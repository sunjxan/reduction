#include "common.hpp"

// Kahan's Summation Formula 算法
// 一种用于减少浮点数加法运算中累积舍入误差的算法。
// 该算法通过维护一个补偿变量c来减少误差，使得求和的结果更加精确。

void reduce(const real *A, size_t size, real *B)
{
    real sum = 0.0, c = 0.0;
    for (size_t i = 0; i < size; ++i) {
        real y = A[i] - c;
        real t = sum + y;
        c = t - sum - y;
        sum = t;
    }
    B[0] = sum - c;
}

int main()
{
    launch_cpu();
    return 0;
}
