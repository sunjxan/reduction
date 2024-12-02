#include "common.hpp"

__global__ void kernel(const real *A, size_t size, real *B)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(B, A[idx]);
    }
}

void reduce(const real *A, size_t size, real *B)
{
    unsigned block_size = 1024, grid_size = DIVUP(size, block_size);
    kernel<<<grid_size, block_size>>>(A, size, B);
    CHECK(cudaDeviceSynchronize());
}

int main()
{
    launch_gpu();
    return 0;
}