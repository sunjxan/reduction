#include "common.hpp"

// 递归折半，顺序配对

real calc(real *A, size_t size)
{
    if (!size) {
        return 0.0;
    }
    if (size == 1) {
        return A[0];
    }
    size_t stride = (size + 1) >> 1;
    for (size_t i = 0; i < stride; ++i) {
        if (i + stride < size) {
            A[i] += A[i + stride];
        }
    }
    return calc(A, stride);
}

void reduce(const real *A, size_t size, real *result)
{
    real *B = nullptr;
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