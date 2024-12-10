#include <cooperative_groups.h>

#include "common.hpp"

using namespace cooperative_groups;

// 使用协作组

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

    for (size_t stride = bdx >> 1; stride > 32; stride >>= 1) {
        if (tid < stride) {
            s_a[tid] += s_a[tid + stride];
        }
        __syncthreads();
    }

    if (tid < 32) {
        s_a[tid] += s_a[tid + 32];

        v = s_a[tid];
        thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
        for (size_t stride = 16; stride > 0; stride >>= 1) {
            v += g.shfl_down(v, stride);
        }
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

    kernel<<<grid_size, block_size, block_size * real_size>>>(d_A, size, d_B);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    unsigned block_size2 = 512;
    kernel<<<1, block_size2, block_size2 * real_size>>>(d_B, grid_size, d_B);
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
