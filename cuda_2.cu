#include "common.hpp"

// atomicAdd对double类型的支持需要对应CUDA版本

__global__ void kernel(const real *A, size_t size, real *result)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }
    atomicAdd(result, A[idx]);
}

void reduce(const real *d_A, size_t size, real *h_result)
{
    real *d_result = nullptr;
    CHECK(cudaMalloc(&d_result, real_size));
    CHECK(cudaMemset(d_result, 0, real_size));

    unsigned block_size = 1024, grid_size = DIVUP(size, block_size);
    kernel<<<grid_size, block_size>>>(d_A, size, d_result);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_result, d_result, real_size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_result));
}

int main()
{
    launch_gpu();
    return 0;
}