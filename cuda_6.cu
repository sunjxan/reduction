#include "common.hpp"

// 为了代替最后传回host计算的步骤，调用两次kernel函数，直接归约成一个数
// 受限于block_size<=1024，每个线程折半之前先累加times倍范围的元素
// 为了第二个kernel可以一个block完成计算，需要调整times值
// 使用共享内存加速

__global__ void kernel(const real *A, size_t size, real *B)
{
    unsigned tid = threadIdx.x, bid = blockIdx.x, bdx = blockDim.x, idx = bid * bdx + tid;
    extern __shared__ real s_a[];
    if (idx >= size) {
        s_a[tid] = 0.0;
        return;
    }

    size_t pos = idx, thread_count = gridDim.x * blockDim.x;
    real v = A[pos];
    while (pos + thread_count < size) {
        pos += thread_count;
        v += A[pos];
    }
    s_a[tid] = v;
    __syncthreads();

    for (size_t stride = bdx >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_a[tid] += s_a[tid + stride];
        }
        __syncthreads();
    }

    if (!tid) {
        B[bid] = s_a[0];
    }
}

void reduce(const real *d_A, size_t size, real *h_result)
{
    if (!size) {
        *h_result = 0.0;
        return;
    }

    // 以1:times的比例估算需要的线程数groups，block_size应是2的整数幂
    unsigned times = 10, groups = DIVUP(size, times), block_size = 1024, grid_size = DIVUP(groups, block_size);
    size_t B_size = grid_size * real_size;

    real *d_B = nullptr;
    CHECK(cudaMalloc(&d_B, B_size));

    kernel<<<grid_size, block_size, block_size * real_size>>>(d_A, size, d_B);
    CHECK(cudaDeviceSynchronize());

    // 保证grid_size=1,block_size=1024的kernel能完成全部计算
    real *d_result = nullptr;
    CHECK(cudaMalloc(&d_result, real_size));

    kernel<<<1, block_size, block_size * real_size>>>(d_B, grid_size, d_result);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_result, d_result, real_size, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_result));
    CHECK(cudaFree(d_B));
}

int main()
{
    launch_gpu();
    return 0;
}
