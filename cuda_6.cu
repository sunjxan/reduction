#include "common.hpp"

// 为了代替最后传回host计算的步骤，调用两次kernel函数，直接归约成一个数
// 受限于block_size<=1024，每个线程折半之前先累加times倍范围的元素
// 为了第二个kernel可以一个block完成计算，需要调整times值

__global__ void kernel(const real *A, size_t size, real *B, size_t thread_count, real *C, unsigned times)
{
    unsigned tid = threadIdx.x, bid = blockIdx.x, bdx = blockDim.x, idx = bid * bdx + tid;
    if (idx >= thread_count) {
        return;
    }

    real v = A[idx];
    for (size_t i = 1; i < times; ++i) {
        if (idx + i * thread_count < size) {
            v += A[idx + i * thread_count];
        } else {
            break;
        }
    }
    B[idx] = v;
    __syncthreads();

    real *Bx = B + bid * bdx;
    for (size_t last_stride = bdx; last_stride > 1; ) {
        size_t stride = (last_stride + 1) >> 1;
        if (tid + stride < last_stride && idx + stride < thread_count) {
            Bx[tid] += Bx[tid + stride];
        }
        last_stride = stride;
        __syncthreads();
    }

    if (!tid) {
        C[bid] = Bx[0];
    }
}

void reduce(const real *d_A, size_t size, real *h_result)
{
    unsigned times = 10;
    size_t thread_count = DIVUP(size, times);
    real *d_B = nullptr;
    CHECK(cudaMalloc(&d_B, thread_count * real_size));

    unsigned block_size = 1024, grid_size = DIVUP(thread_count, block_size);
    size_t total_size = grid_size * real_size;
    real *d_C = nullptr, *h_C = nullptr;
    CHECK(cudaMalloc(&d_C, total_size));
    CHECK(cudaMallocHost(&h_C, total_size));

    kernel<<<grid_size, block_size>>>(d_A, size, d_B, thread_count, d_C, times);
    CHECK(cudaDeviceSynchronize());

    // 保证grid_size=1,block_size=1024的kernel能完成全部计算
    times = DIVUP(grid_size, block_size);
    thread_count = DIVUP(grid_size, times);
    real *d_result = nullptr;
    CHECK(cudaMalloc(&d_result, real_size));

    kernel<<<1, block_size>>>(d_C, grid_size, d_B, thread_count, d_result, times);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_result, d_result, real_size, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_result));
    CHECK(cudaFreeHost(h_C));
    CHECK(cudaFree(d_C));
    CHECK(cudaFree(d_B));
}

int main()
{
    launch_gpu();
    return 0;
}
