#include "common.hpp"

// 递归二分，相邻配对

real calc(const real *A, size_t l, size_t r)
{
    if (l >= r) {
        return 0;
    }
    size_t mid = (l + r) >> 1;
    if (l == mid) {
        return A[l];
    }
    return calc(A, l, mid) + calc(A, mid, r);
}

void reduce(const real *A, size_t size, real *B)
{
    B[0] = calc(A, 0, size);
}

int main()
{
    launch_cpu();
    return 0;
}
