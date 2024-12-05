#pragma once
#include <cstdio>
#include <cmath>

#include "error.h"

constexpr unsigned SKIP = 5, REPEATS = 5;
constexpr size_t N = 1e8 + 7;
constexpr size_t real_size = sizeof(real);
constexpr size_t N_size = N * real_size;

void reduce(const real *, const size_t, real *);

void random_init(real *data, const size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        data[i] = real(rand()) / RAND_MAX;
    }
}

// Kahan's Summation Formula 算法
// 一种用于减少浮点数加法运算中累积舍入误差的算法。
// 该算法通过维护一个补偿变量c来减少误差，使得求和的结果更加精确。
real get_answer(const real *A, size_t size)
{
    real sum = 0.0, c = 0.0;
    for (size_t i = 0; i < size; ++i) {
        real y = A[i] - c;
        real t = sum + y;
        c = t - sum - y;
        sum = t;
    }
    return sum - c;
}

real timing(const real *A, const size_t size, real *result)
{
    float elapsed_time = 0;
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start, 0));

    reduce(A, size, result);

    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return elapsed_time;
}

void launch_cpu()
{
    real *h_A = nullptr, result = 0.0;
    CHECK(cudaMallocHost(&h_A, N_size));

    random_init(h_A, N);

    float elapsed_time = 0, total_time = 0;
    for (unsigned i = 0; i < SKIP; ++i) {
        elapsed_time = timing(h_A, N, &result);
    }
    for (unsigned i = 0; i < REPEATS; ++i) {
        elapsed_time = timing(h_A, N, &result);
        total_time += elapsed_time;
    }
    printf("Time: %9.3f ms\n", total_time / REPEATS);

    real answer = get_answer(h_A, N);
    real absolute_error = fabs(result - answer), relative_error = absolute_error / answer * 100;
    printf("Result: %16.6f  Answer: %16.6f  Error: %16.6f %6.2f%%\n", result, answer, absolute_error, relative_error);

    CHECK(cudaFreeHost(h_A));
}

void launch_gpu()
{
    real *h_A = nullptr, result = 0.0;
    CHECK(cudaMallocHost(&h_A, N_size));

    random_init(h_A, N);

    real *d_A = nullptr;
    CHECK(cudaMalloc(&d_A, N_size));

    CHECK(cudaMemcpy(d_A, h_A, N_size, cudaMemcpyHostToDevice));

    float elapsed_time = 0, total_time = 0;
    for (unsigned i = 0; i < SKIP; ++i) {
        elapsed_time = timing(d_A, N, &result);
    }
    for (unsigned i = 0; i < REPEATS; ++i) {
        elapsed_time = timing(d_A, N, &result);
        total_time += elapsed_time;
    }
    printf("Time: %9.3f ms\n", total_time / REPEATS);

    real answer = get_answer(h_A, N);
    real absolute_error = fabs(result - answer), relative_error = absolute_error / answer * 100;
    printf("Result: %16.6f  Answer: %16.6f  Error: %16.6f %6.2f%%\n", result, answer, absolute_error, relative_error);

    CHECK(cudaFree(d_A));
    CHECK(cudaFreeHost(h_A));
}
