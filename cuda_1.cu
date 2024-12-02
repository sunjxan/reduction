
#include "common.hpp"

__global__ void kernel(const real *A, size_t size, real *B)
{
    real sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum += A[i];
    }
    B[0] = sum;
}

void reduce(const real *A, size_t size, real *B)
{
    kernel<<<1, 1>>>(A, size, B);
    CHECK(cudaDeviceSynchronize());
}

int main()
{
    launch_gpu();
    return 0;
}