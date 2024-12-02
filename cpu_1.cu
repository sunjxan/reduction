#include "common.hpp"

void reduce(const real *A, const size_t size, real *B)
{
    real sum = 0;
    for (size_t i = 0; i < size; ++i) {
        sum += A[i];
    }
    B[0] = sum;
}

int main()
{
    launch_cpu();
    return 0;
}
