#include "common.hpp"

// atomicAdd对double类型的支持需要对应CUDA版本

__global__ void kernel(const real *A, size_t size, real *result)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(result, A[idx]);
    }
}

void reduce(const real *A, size_t size, real *result)
{
    real *d_result;
    CHECK(cudaMalloc(&d_result, real_size));

    unsigned block_size = 1024, grid_size = DIVUP(size, block_size);
    kernel<<<grid_size, block_size>>>(A, size, d_result);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(result, d_result, real_size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_result));
}

int main()
{
    launch_gpu();
    return 0;
}