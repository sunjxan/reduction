#include <cmath>

#include "common.hpp"

// 分多组，分别求和再计算总和

void reduce(const real *A, size_t size, real *result)
{
    unsigned stride = ceil(sqrt(size));

    real sum = 0.0;
    for (size_t i = 0; i < stride; ++i) {
        real v = 0.0;
        for (size_t j = i; j < size; j += stride) {
            v += A[j];
        }
        sum += v;
    }

    *result = sum;
}

int main()
{
    launch_cpu();
    return 0;
}