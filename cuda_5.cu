#include "common.hpp"

// 每个block计算一个部分和，相邻配对
// 使用共享内存加速
// 始终让threadIdx.x最小的部分线程工作，减少Warp分化的影响

__global__ void kernel(const real *A, size_t size, real *B)
{
    unsigned tid = threadIdx.x, bid = blockIdx.x, bdx = blockDim.x, idx = bid * bdx + tid;
    extern __shared__ real s_a[];
    if (idx >= size) {
        s_a[tid] = 0.0;
        return;
    }

    size_t pos = idx * 2;
    real v = A[pos];
    if (pos + 1 < size) {
        v += A[pos + 1];
    }
    s_a[tid] = v;
    __syncthreads();

    for (size_t stride = 1; stride < bdx; stride <<= 1) {
        size_t index = tid * (stride << 1);
        if (index < bdx) {
            s_a[index] += s_a[index + stride];
        }
        __syncthreads();
    }

    if (!tid) {
        B[bid] = s_a[0];
    }
}

void reduce(const real *d_A, size_t size, real *h_result)
{
    // 以1:2的比例估算需要的线程数groups，block_size应是2的整数幂
    unsigned groups = DIVUP(size, 2), block_size = 1024, grid_size = DIVUP(groups, block_size);
    size_t B_size = grid_size * real_size;

    real *d_B = nullptr, *h_B = nullptr;
    CHECK(cudaMalloc(&d_B, B_size));
    CHECK(cudaMallocHost(&h_B, B_size));
    CHECK(cudaMemset(d_B, 0, B_size));

    kernel<<<grid_size, block_size, block_size * real_size>>>(d_A, size, d_B);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_B, d_B, B_size, cudaMemcpyDeviceToHost));
    real sum = 0.0;
    for (size_t i = 0; i < grid_size; ++i) {
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