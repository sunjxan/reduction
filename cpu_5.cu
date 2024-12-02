#include "common.hpp"

// 分两段交错配对

void reduce(const real *A, size_t size, real *B)
{
    size_t offset = 1;
    while (offset < size) {
        offset <<= 1;
    }
    offset >>= 1;

    real *C;
    CHECK(cudaMallocHost(&C, offset * real_size));

    bool first = true;
    while (offset) {
        for (size_t pos = 0; pos < offset; ++pos) {
            size_t target = pos + offset;
            if (first) {
                C[pos] = A[pos];
                if (target < size) {
                    C[pos] += A[target];
                }
            } else if (target < size) {
                C[pos] += C[target];
            }
        }
        offset >>= 1;
        first = false;
    }
    B[0] = C[0];

    CHECK(cudaFreeHost(C));
}

int main()
{
    launch_cpu();
    return 0;
}
