#include "common.hpp"

// 迭代二分，相邻配对

void reduce(const real *A, size_t size, real *B)
{
    real sum = 0.0;
    size_t l = 0, r = size;
    size_t ls[30], rs[30], top = 0, last_l = 0;
    real vs[30];
    while (true) {
        size_t mid = (l + r) >> 1;
        if (top && ls[top - 1] == l && rs[top - 1] == r) {
            if (l == last_l) {
                l = mid;
            } else {
                --top;
                if (!top) {
                    sum = vs[0];
                    break;
                }
                last_l = l;
                l = ls[top - 1];
                r = rs[top - 1];
                if (l == last_l) {
                    vs[top - 1] = vs[top];
                } else {
                    vs[top - 1] += vs[top];
                }
            }
            continue;
        }
        if (l == mid) {
            if (!top) {
                sum = A[l];
                break;
            }
            last_l = l;
            l = ls[top - 1];
            r = rs[top - 1];
            if (l == last_l) {
                vs[top - 1] = A[l];
            } else {
                vs[top - 1] += A[l];
            }
            continue;
        }
        ls[top] = l;
        rs[top] = r;
        vs[top] = 0;
        ++top;
        r = mid;
    }
    B[0] = sum;
}

int main()
{
    launch_cpu();
    return 0;
}
