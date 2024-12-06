#include "common.hpp"

// 分多组，每个线程计算一个组的和

__global__ void kernel(const real *A, size_t size, real *B)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }
    real sum = 0.0;
    for (size_t i = idx; i < size; i += gridDim.x * blockDim.x) {
        sum += A[i];
    }
    B[idx] = sum;
}

void reduce(const real *d_A, size_t size, real *h_result)
{
    // 估算数组B的长度和需要的线程数groups，thread_count是实际使用的线程数
    unsigned groups = DIVUP(size, 128), block_size = 256, grid_size = DIVUP(groups, block_size);
    size_t thread_count = grid_size * block_size, B_size = thread_count * real_size;
    real *d_B = nullptr, *h_B = nullptr;
    CHECK(cudaMalloc(&d_B, B_size));
    CHECK(cudaMallocHost(&h_B, B_size));
    CHECK(cudaMemset(d_B, 0, B_size));
    
    kernel<<<grid_size, block_size>>>(d_A, size, d_B);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_B, d_B, B_size, cudaMemcpyDeviceToHost));
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
