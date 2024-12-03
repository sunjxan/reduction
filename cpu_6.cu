#include "common.hpp"

// 迭代折半，交错配对

void reduce(const real *A, size_t size, real *result)
{
    real *B;
    size_t total_size = size * real_size;
    CHECK(cudaMallocHost(&B, total_size));
    CHECK(cudaMemcpy(B, A, total_size, cudaMemcpyHostToHost));

    size_t stride = (size + 1) >> 1;
    while (size > 1) {
        for (size_t i = 0; i < stride; ++i) {
            size_t target = i + stride;
            if (target < size) {
                B[i] += B[target];
            }
        }
        size = stride;
        stride = (size + 1) >> 1;
    }
    *result = B[0];

    CHECK(cudaFreeHost(B));
}

int main()
{
    launch_cpu();
    return 0;
}
