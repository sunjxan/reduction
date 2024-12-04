#include "common.hpp"

// 每个block计算一个部分和，交错配对

__global__ void kernel(const real *A, size_t size, real *B, size_t thread_count, real *C)
{
    unsigned tid = threadIdx.x, bid = blockIdx.x, bdx = blockDim.x, idx = bid * bdx + tid;
    if (idx >= thread_count) {
        return;
    }

    real v = A[idx];
    if (idx + thread_count < size) {
        v += A[idx + thread_count];
    }
    B[idx] = v;
    __syncthreads();

    real *Bx = B + bid * bdx;
    for (size_t last_stride = bdx; last_stride > 1; ) {
        size_t stride = (last_stride + 1) >> 1;
        if (tid < stride && tid + stride < last_stride && idx + stride < thread_count) {
            Bx[tid] += Bx[tid + stride];
        }
        last_stride = stride;
        __syncthreads();
    }

    if (!tid) {
        C[bid] = Bx[0];
    }
}

void reduce(const real *d_A, size_t size, real *h_result)
{
    size_t thread_count = (size + 1) >> 1;
    real *d_B = nullptr;
    CHECK(cudaMalloc(&d_B, thread_count * real_size));

    unsigned block_size = 1024, grid_size = DIVUP(thread_count, block_size);
    size_t total_size = grid_size * real_size;
    real *d_C = nullptr, *h_C = nullptr;
    CHECK(cudaMalloc(&d_C, total_size));
    CHECK(cudaMallocHost(&h_C, total_size));

    kernel<<<grid_size, block_size>>>(d_A, size, d_B, thread_count, d_C);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_C, d_C, total_size, cudaMemcpyDeviceToHost));
    real sum = 0.0;
    for (size_t i = 0; i < grid_size; ++i) {
        sum += h_C[i];
    }
    *h_result = sum;

    CHECK(cudaFreeHost(h_C));
    CHECK(cudaFree(d_C));
    CHECK(cudaFree(d_B));
}

int main()
{
    launch_gpu();
    return 0;
}