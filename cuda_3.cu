#include "common.hpp"

__global__ void kernel(const real *A, size_t size, real *B, size_t group_count, size_t group_size)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < group_count) {
        unsigned beg = idx * group_size, end = beg + group_size;
        if (end > size) {
            end = size;
        }
        real sum = 0.0;
        for (size_t i = beg; i < end; ++i) {
            sum += A[i];
        }
        B[idx] = sum;
    }
}

void reduce(const real *d_A, size_t size, real *h_result)
{
    const size_t group_count = 1e6, group_size = DIVUP(size, group_count), total_size = group_count * real_size;

    real *d_B, *h_B;
    CHECK(cudaMalloc(&d_B, total_size));
    CHECK(cudaMallocHost(&h_B, total_size));

    unsigned block_size = 128, grid_size = DIVUP(group_count, block_size);
    kernel<<<grid_size, block_size>>>(d_A, size, d_B, group_count, group_size);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_B, d_B, total_size, cudaMemcpyDeviceToHost));

    real sum = 0.0;
    for (size_t i = 0; i < group_count; ++i) {
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