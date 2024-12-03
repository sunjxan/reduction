#include "common.hpp"

// 迭代折半，交错配对

void reduce(const real *A, size_t size, real *result)
{
    real *B = nullptr;
    size_t total_size = size * real_size;
    CHECK(cudaMallocHost(&B, total_size));
    CHECK(cudaMemcpy(B, A, total_size, cudaMemcpyHostToHost));

    for (size_t last_stride = size; last_stride > 1; ) {
        size_t stride = (last_stride + 1) >> 1;
        for (size_t i = 0; i < stride; ++i) {
            if (i + stride < last_stride) {
                B[i] += B[i + stride];
            }
        }
        last_stride = stride;
    }
    *result = B[0];

    CHECK(cudaFreeHost(B));
}

int main()
{
    launch_cpu();
    return 0;
}
