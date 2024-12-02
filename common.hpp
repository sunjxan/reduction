#pragma once
#include <cstdio>

#include "error.h"

constexpr unsigned SKIP = 5, REPEATS = 5;
constexpr size_t N = 1e8 + 7;
constexpr size_t real_size = sizeof(real);
constexpr size_t N_size = N * real_size;
constexpr real element = 1.23;

void reduce(const real *, const size_t, real *);

void init(real *data, const size_t size, const real value)
{
    for (size_t i = 0; i < size; ++i) {
        data[i] = value;
    }
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

    init(h_A, N, element);

    float elapsed_time = 0, total_time = 0;
    for (unsigned i = 0; i < SKIP; ++i) {
        elapsed_time = timing(h_A, N, &result);
    }
    for (unsigned i = 0; i < REPEATS; ++i) {
        elapsed_time = timing(h_A, N, &result);
        total_time += elapsed_time;
    }
    printf("Time: %9.3f ms\n", total_time / REPEATS);

    const real answer = element * N;
    real absolute_error = fabs(result - answer), relative_error = absolute_error / answer * 100;
    printf("Result: %16.6f  Answer: %16.6f  Error: %16.6f %6.2f%%\n", result, answer, absolute_error, relative_error);

    CHECK(cudaFreeHost(h_A));
}

void launch_gpu()
{
    real *h_A = nullptr, result = 0.0;
    CHECK(cudaMallocHost(&h_A, N_size));

    init(h_A, N, element);

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

    const real answer = element * N;
    real absolute_error = fabs(result - answer), relative_error = absolute_error / answer * 100;
    printf("Result: %16.6f  Answer: %16.6f  Error: %16.6f %6.2f%%\n", result, answer, absolute_error, relative_error);

    CHECK(cudaFree(d_A));
    CHECK(cudaFreeHost(h_A));
}
