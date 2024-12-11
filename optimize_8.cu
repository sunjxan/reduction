#include <cooperative_groups.h>

#include "common.hpp"

using namespace cooperative_groups;

// 使用模板核函数，指定block大小作为模板参数

template<unsigned blockSize>
__global__ void kernel(const real *A, size_t size, real *B)
{
    unsigned tid = threadIdx.x, bid = blockIdx.x, bdx = blockDim.x, idx = bid * bdx + tid;
    extern __shared__ real s_a[];
    if (idx >= size) {
        s_a[tid] = 0.0;
        return;
    }

    size_t pos = idx, thread_count = gridDim.x * blockDim.x;
    real v = A[pos];
    while (pos + thread_count < size) {
        pos += thread_count;
        v += A[pos];
    }
    s_a[tid] = v;
    __syncthreads();

    if (blockSize >= 1024) {
        if (tid < 512) {
            s_a[tid] += s_a[tid + 512];
        }
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) {
            s_a[tid] += s_a[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            s_a[tid] += s_a[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            s_a[tid] += s_a[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) {
        s_a[tid] += s_a[tid + 32];

        v = s_a[tid];
        thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
        v += g.shfl_down(v, 16);
        v += g.shfl_down(v, 8);
        v += g.shfl_down(v, 4);
        v += g.shfl_down(v, 2);
        v += g.shfl_down(v, 1);
    }

    if (!tid) {
        B[bid] = v;
    }
}

void reduce(const real *d_A, size_t size, real *h_result)
{
    // 以1:times的比例估算需要的线程数groups，block_size应是2的整数幂，且大于等于64
    unsigned times = 10, groups = DIVUP(size, times), block_size = 1024, grid_size = DIVUP(groups, block_size);
    size_t B_size = grid_size * real_size;

    real *d_B = nullptr;
    CHECK(cudaMalloc(&d_B, B_size));

    switch(block_size) {
        case 1024:
            kernel<1024><<<grid_size, block_size, block_size * real_size>>>(d_A, size, d_B);
            break;
        case 512:
            kernel<512><<<grid_size, block_size, block_size * real_size>>>(d_A, size, d_B);
            break;
        case 256:
            kernel<256><<<grid_size, block_size, block_size * real_size>>>(d_A, size, d_B);
            break;
        case 128:
            kernel<128><<<grid_size, block_size, block_size * real_size>>>(d_A, size, d_B);
            break;
        case 64:
            kernel<64><<<grid_size, block_size, block_size * real_size>>>(d_A, size, d_B);
            break;
    }
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    unsigned block_size2 = 512;
    switch(block_size2) {
        case 1024:
            kernel<1024><<<1, block_size2, block_size2 * real_size>>>(d_B, grid_size, d_B);
            break;
        case 512:
            kernel<512><<<1, block_size2, block_size2 * real_size>>>(d_B, grid_size, d_B);
            break;
        case 256:
            kernel<256><<<1, block_size2, block_size2 * real_size>>>(d_B, grid_size, d_B);
            break;
        case 128:
            kernel<128><<<1, block_size2, block_size2 * real_size>>>(d_B, grid_size, d_B);
            break;
        case 64:
            kernel<64><<<1, block_size2, block_size2 * real_size>>>(d_B, grid_size, d_B);
            break;
    }
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_result, d_B, real_size, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_B));
}

int main()
{
    launch_gpu();
    return 0;
}