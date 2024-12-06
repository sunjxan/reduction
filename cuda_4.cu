#include "common.hpp"

// 每个block计算一个部分和，相邻配对

__global__ void kernel(const real *A, size_t size, real *B, size_t thread_count, real *C)
{
    unsigned tid = threadIdx.x, bid = blockIdx.x, bdx = blockDim.x, idx = bid * bdx + tid;
    if (idx >= size) {
        return;
    }

    size_t pos = idx * 2;
    real v = A[pos];
    if (pos + 1 < size) {
        v += A[pos + 1];
    }
    B[idx] = v;
    __syncthreads();

    real *Bx = B + bid * bdx;
    for (size_t stride = 1; stride < bdx; stride <<= 1) {
        if (!(tid % (stride << 1))) {
            Bx[tid] += Bx[tid + stride];
        }
        __syncthreads();
    }

    if (!tid) {
        C[bid] = Bx[0];
    }
}

void reduce(const real *d_A, size_t size, real *h_result)
{
    // 以1:2的比例估算数组B的长度和需要的线程数groups，block_size应是2的整数幂，thread_count是实际使用的线程数
    unsigned groups = DIVUP(size, 2), block_size = 1024, grid_size = DIVUP(groups, block_size);
    size_t thread_count = grid_size * block_size, B_size = thread_count * real_size, C_size = grid_size * real_size;

    real *d_B = nullptr;
    CHECK(cudaMalloc(&d_B, B_size));
    // 为折半设置初值0
    CHECK(cudaMemset(d_B, 0, B_size));

    real *d_C = nullptr, *h_C = nullptr;
    CHECK(cudaMalloc(&d_C, C_size));
    CHECK(cudaMallocHost(&h_C, C_size));

    kernel<<<grid_size, block_size>>>(d_A, size, d_B, thread_count, d_C);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_C, d_C, C_size, cudaMemcpyDeviceToHost));
    real sum = 0.0;
    for (size_t i = 0; i < grid_size; ++i) {
        sum += h_C[i];
    }
    *h_result = sum;

    CHECK(cudaFreeHost(h_C));
    CHECK(cudaFree(d_C));
    CHECK(cudaFree(d_B));
}

int main()
{
    launch_gpu();
    return 0;
}