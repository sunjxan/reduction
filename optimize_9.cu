#include <cooperative_groups.h>

#include "common.hpp"

using namespace cooperative_groups;

// 使用模板核函数，指定block大小作为模板参数

template<unsigned blockSize>
__device__ __forceinline__ real warpReduce(real v, thread_block_tile<32> g) {
    if (blockSize >= 32) {
        v += g.shfl_down(v, 16);
    }
    if (blockSize >= 16) {
        v += g.shfl_down(v, 8);
    }
    if (blockSize >= 8) {
        v += g.shfl_down(v, 4);
    }
    if (blockSize >= 4) {
        v += g.shfl_down(v, 2);
    }
    if (blockSize >= 2) {
        v += g.shfl_down(v, 1);
    }
    return v;
}

template<unsigned blockSize>
__global__ void kernel(const real *A, size_t size, real *B)
{
    unsigned tid = threadIdx.x, bid = blockIdx.x, bdx = blockDim.x, idx = bid * bdx + tid;
    unsigned laneIdx = tid % warpSize, warpIdx = tid / warpSize;
    const unsigned warp_count = DIVUP(blockSize, 32);
    extern __shared__ real s_a[];
    if (idx >= size) {
        return;
    }

    size_t pos = idx, thread_count = gridDim.x * blockDim.x;
    real v = A[pos];
    while (pos + thread_count < size) {
        pos += thread_count;
        v += A[pos];
    }

    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
    v = warpReduce<blockSize>(v, g);

    if (!laneIdx) {
        s_a[warpIdx] = v;
    }
    __syncthreads();

    if (tid < warp_count) {
        v = warpReduce<warp_count>(s_a[laneIdx], g);
    }

    if (!tid) {
        B[bid] = v;
    }
}

void reduce(const real *d_A, size_t size, real *h_result)
{
    // 以1:times的比例估算需要的线程数groups，block_size应是2的整数幂，且大于等于32
    unsigned times = 10, groups = DIVUP(size, times), block_size = 1024, grid_size = DIVUP(groups, block_size);
    size_t B_size = grid_size * real_size;

    real *d_B = nullptr;
    CHECK(cudaMalloc(&d_B, B_size));

    switch(block_size) {
        case 1024:
            kernel<1024><<<grid_size, block_size, DIVUP(block_size, 32) * real_size>>>(d_A, size, d_B);
            break;
        case 512:
            kernel<512><<<grid_size, block_size, DIVUP(block_size, 32) * real_size>>>(d_A, size, d_B);
            break;
        case 256:
            kernel<256><<<grid_size, block_size, DIVUP(block_size, 32) * real_size>>>(d_A, size, d_B);
            break;
        case 128:
            kernel<128><<<grid_size, block_size, DIVUP(block_size, 32) * real_size>>>(d_A, size, d_B);
            break;
        case 64:
            kernel<64><<<grid_size, block_size, DIVUP(block_size, 32) * real_size>>>(d_A, size, d_B);
            break;
        case 32:
            kernel<32><<<grid_size, block_size, DIVUP(block_size, 32) * real_size>>>(d_A, size, d_B);
            break;
    }
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    unsigned block_size2 = 512;
    switch(block_size2) {
        case 1024:
            kernel<1024><<<1, block_size2, DIVUP(block_size2, 32) * real_size>>>(d_B, grid_size, d_B);
            break;
        case 512:
            kernel<512><<<1, block_size2, DIVUP(block_size2, 32) * real_size>>>(d_B, grid_size, d_B);
            break;
        case 256:
            kernel<256><<<1, block_size2, DIVUP(block_size2, 32) * real_size>>>(d_B, grid_size, d_B);
            break;
        case 128:
            kernel<128><<<1, block_size2, DIVUP(block_size2, 32) * real_size>>>(d_B, grid_size, d_B);
            break;
        case 64:
            kernel<64><<<1, block_size2, DIVUP(block_size2, 32) * real_size>>>(d_B, grid_size, d_B);
            break;
        case 32:
            kernel<32><<<1, block_size2, DIVUP(block_size2, 32) * real_size>>>(d_B, grid_size, d_B);
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