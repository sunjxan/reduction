#include <cooperative_groups.h>

#include "common.hpp"

using namespace cooperative_groups;

// 一个核函数中，两次在Warp内使用协作组做顺序配对的折半归约

__device__ __forceinline__ real warpReduce(real v, thread_block_tile<32> g) {
    v += g.shfl_down(v, 16);
    v += g.shfl_down(v, 8);
    v += g.shfl_down(v, 4);
    v += g.shfl_down(v, 2);
    v += g.shfl_down(v, 1);
    return v;
}

__global__ void kernel(const real *A, size_t size, real *B)
{
    unsigned tid = threadIdx.x, bid = blockIdx.x, bdx = blockDim.x, idx = bid * bdx + tid;
    unsigned laneIdx = tid % warpSize, warpIdx = tid / warpSize, warp_count = DIVUP(bdx, warpSize);
    extern __shared__ real s_a[];
    if (idx >= size) {
        if (!laneIdx) {
            s_a[warpIdx] = 0.0;
        }
        return;
    }

    size_t pos = idx, thread_count = gridDim.x * blockDim.x;
    real v = A[pos];
    while (pos + thread_count < size) {
        pos += thread_count;
        v += A[pos];
    }

    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
    v = warpReduce(v, g);

    if (!laneIdx) {
        s_a[warpIdx] = v;
    }
    __syncthreads();

    if (tid < warpSize) {
        v = tid < warp_count ? s_a[laneIdx] : 0.0;
        v = warpReduce(v, g);
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

    kernel<<<grid_size, block_size, DIVUP(block_size, 32) * real_size>>>(d_A, size, d_B);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    unsigned block_size2 = 512;
    kernel<<<1, block_size2, DIVUP(block_size2, 32) * real_size>>>(d_B, grid_size, d_B);
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