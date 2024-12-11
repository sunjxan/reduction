#include "common.hpp"

// 迭代二分，相邻配对

void reduce(const real *A, size_t size, real *result)
{
    size_t l = 0, r = size;
    // top - 1 是栈顶元素指针
    size_t ls[30], rs[30], top = 0, last_l = 0;
    real vs[30];
    while (true) {
        size_t mid = (l + r) >> 1;
        if (top && ls[top - 1] == l && rs[top - 1] == r) {
            // lr匹配，不是第一次到达该结点
            if (l == last_l) {
                // 是从左分支回溯，下一步进入右分支
                l = mid;
            } else {
                // 是从右分支回溯，下一步回溯
                real v = vs[top - 1];
                --top;
                // 如果父结点是根结点，退出
                if (!top) {
                    break;
                }
                // 准备回溯后的上下文
                last_l = l;
                l = ls[top - 1];
                r = rs[top - 1];
                if (l == last_l) {
                    // 以左分支身份回溯
                    vs[top - 1] = v;
                } else {
                    // 以右分支身份回溯
                    vs[top - 1] += v;
                }
            }
        } else if (l >= r) {
            if (!top) {
                vs[0] = 0.0;
                break;
            }
            last_l = l;
            l = ls[top - 1];
            r = rs[top - 1];
        } else if (l + 1 == r) {
            // 第一次到达该结点
            real v = A[l];
            // 结点只包含一个元素，没有入栈不用递减top
            if (!top) {
                // 如果该结点是根结点，退出
                vs[0] = v;
                break;
            }
            // 下一步回溯，准备回溯后的上下文
            last_l = l;
            l = ls[top - 1];
            r = rs[top - 1];
            if (l == last_l) {
                // 以左分支身份回溯
                vs[top - 1] = v;
            } else {
                // 以右分支身份回溯
                vs[top - 1] += v;
            }
        } else {
            // 第一次到达该结点
            // 分左右分支
            ls[top] = l;
            rs[top] = r;
            ++top;
            // 不回溯不用设置last_l
            // 下一步进入左分支
            r = mid;
        }
    }
    *result = vs[0];
}

int main()
{
    launch_cpu();
    return 0;
}