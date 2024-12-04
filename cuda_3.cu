#include "common.hpp"

// 分多组，每个线程计算一个组的和

__global__ void kernel(const real *A, size_t size, real *B, size_t thread_count)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= thread_count) {
        return;
    }
    real sum = 0.0;
    for (size_t i = idx; i < size; i += thread_count) {
        sum += A[i];
    }
    B[idx] = sum;
}

void reduce(const real *d_A, size_t size, real *h_result)
{
    size_t thread_count = DIVUP(size, 128), total_size = thread_count * real_size;
    real *d_B = nullptr, *h_B = nullptr;
    CHECK(cudaMalloc(&d_B, total_size));
    CHECK(cudaMallocHost(&h_B, total_size));

    unsigned block_size = 256, grid_size = DIVUP(thread_count, block_size);
    kernel<<<grid_size, block_size>>>(d_A, size, d_B, thread_count);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_B, d_B, total_size, cudaMemcpyDeviceToHost));
    real sum = 0.0;
    for (size_t i = 0; i < thread_count; ++i) {
        sum += h_B[i];
    }
    *h_result = sum;

    CHECK(cudaFreeHost(h_B));
    CHECK(cudaFree(d_B));
}

int main()
{
    launch_gpu();
    return 0;
}
