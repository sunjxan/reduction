#include "common.hpp"

__global__ void kernel(const real *A, size_t size, real *result)
{
    real sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum += A[i];
    }
    *result = sum;
}

void reduce(const real *A, size_t size, real *result)
{
    real *d_result;
    CHECK(cudaMalloc(&d_result, real_size));

    kernel<<<1, 1>>>(A, size, d_result);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(result, d_result, real_size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_result));
}

int main()
{
    launch_gpu();
    return 0;
}