#include "common.hpp"

void reduce(const real *A, size_t size, real *result)
{
    real sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum += A[i];
    }
    *result = sum;
}

int main()
{
    launch_cpu();
    return 0;
}
