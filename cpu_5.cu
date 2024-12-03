#include "common.hpp"

// 递归二分，交错配对

real calc(real *A, size_t size)
{
    if (size == 1) {
        return A[0];
    }
    size_t stride = (size + 1) >> 1;
    for (size_t i = 0; i < stride; ++i) {
        size_t target = i + stride;
        if (target < size) {
            A[i] += A[target];
        }
    }
    return calc(A, stride);
}

void reduce(const real *A, size_t size, real *result)
{
    real *B;
    size_t total_size = size * real_size;
    CHECK(cudaMallocHost(&B, total_size));
    CHECK(cudaMemcpy(B, A, total_size, cudaMemcpyHostToHost));
    *result = calc(B, size);
    CHECK(cudaFreeHost(B));
}

int main()
{
    launch_cpu();
    return 0;
}
