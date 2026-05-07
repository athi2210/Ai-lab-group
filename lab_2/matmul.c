/*
 * Prakitikum 1 – Matrix Multiplication on CPU
 * ============================================
 * AI Accelerators (AIA) – Lab Assignment
 *
 * Your task is to progressively optimize this naive C implementation
 * of matrix multiplication (C = A * B) through the steps below.
 * Read README.md carefully before you start!
 *
 * Build:  make
 * Run:    ./matmul <size>      (e.g. ./matmul 512)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tgmath.h>
#include <time.h>
#include <pthread.h>

#define num_threads 8
const int num_iterations = 4;
#define JB 128 // Tile size divides matrix size

// ============================================================================
// IMPLEMENTATION 1: NAIVE MATRIX MULTIPLICATION
// ============================================================================
void matmul_naive(const float *A, const float *B, float *C, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0;
            for (int k = 0; k < K; k++)
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// IMPLEMENTATION 2: -03 -ffmastmath, does loop unrolling and vectorization, reorder k,j
// ============================================================================
void matmul_looporder(const float *A, const float *B, float *C, int M, int N, int K)
{
    // i = rows_a, j = cols_b, k = place
    // ikj order: better cache use for B and C (row-major layout)

    memset(C, 0, (size_t)M * N * sizeof(float));
    for (int i = 0; i < M; i++)
    {
        for (int k = 0; k < K; k++)
        {
            for (int j = 0; j < N; j++)
            {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

// ============================================================================
// IMPLEMENTATION 3: Tiling
// ============================================================================
void sliced_mul(const float *A, const float *B, float *C, int M, int N, int K, int ri, int ci, int ki)
{
    // i = rows_a, j = cols_b, k = place
    int r, c, p;
    for (int i = 0; i < JB; i++)
    {
        r = ri * JB + i;
        for (int k = 0; k < JB; k++)
        {
            for (int j = 0; j < JB; j++)
            {
                c = ci * JB + j;
                C[r * N + c] += A[r * K + (ki * JB + k)] * B[(ki * JB + k) * N + c];
            }
        }
    }
}

typedef struct
{
    const float *A;
    const float *B;
    float *C;
    int M;
    int N;
    int K;
    int ri_start;
    int ri_end;
    int num_ci;
    int num_ki;
} worker_args_t;

static void *matmul_worker(void *arg)
{
    worker_args_t *w = (worker_args_t *)arg;

    for (int ri = w->ri_start; ri < w->ri_end; ri++)
    {
        for (int ki = 0; ki < w->num_ki; ki++)
        {
            for (int ci = 0; ci < w->num_ci; ci++)
            {
                sliced_mul(w->A, w->B, w->C, w->M, w->N, w->K, ri, ci, ki);
            }
        }
    }

    return NULL;
}

void matmul_looptiling(const float *A, const float *B, float *C, int M, int N, int K)
{
    /* TODO: implement loop-tiled matrix multiplication */
    memset(C, 0, (size_t)M * N * sizeof(float));

    int num_ri = M / JB;
    int num_ci = N / JB;
    int num_ki = K / JB;

    for (int ri = 0; ri < num_ri; ri++)
    {
        for (int ki = 0; ki < num_ki; ki++)
        {
            for (int ci = 0; ci < num_ci; ci++)

            {
                sliced_mul(A, B, C, M, N, K, ri, ci, ki);
            }
        }
    }
}

// ============================================================================
// IMPLEMENTATION 4: Multithreading
// ============================================================================
void matmul_parallel_ikj(const float *A, const float *B, float *C,
                         int M, int N, int K)
{
    memset(C, 0, (size_t)M * N * sizeof(float));

    int num_ri = M / JB;
    int num_ci = N / JB;
    int num_ki = K / JB;

    pthread_t threads[num_threads];
    worker_args_t args[num_threads];

    int rows_per_thread = num_ri / num_threads;
    int remainder = num_ri % num_threads;
    int next_ri = 0;

    for (int t = 0; t < num_threads; t++)
    {
        int chunk = rows_per_thread + (t < remainder ? 1 : 0);
        args[t].A = A;
        args[t].B = B;
        args[t].C = C;
        args[t].M = M;
        args[t].N = N;
        args[t].K = K;
        args[t].ri_start = next_ri;
        args[t].ri_end = next_ri + chunk;
        args[t].num_ci = num_ci;
        args[t].num_ki = num_ki;

        next_ri += chunk;
        pthread_create(&threads[t], NULL, matmul_worker, &args[t]);
    }

    for (int t = 0; t < num_threads; t++)
    {
        pthread_join(threads[t], NULL);
    }
}

// ============================================================================
// Utility functions: Init Matrix, Benchmarking, Calculate Gflops
// ============================================================================
void initialize_matrix(float *matrix, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
    {
        matrix[i] = rand() % 100;
    }
}

double get_time_ms()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

double calculate_gflops(int M, int N, int K, double total_time)
{
    double flops = 2.0 * M * N * K;
    double gflops = (flops / ((total_time) / 1000.0)) / 1e9;
    return gflops;
}

int verify_result(const float *C_ref, const float *C_test, int M, int N, float tolerance)
{
    for (int i = 0; i < M * N; i++)
    {
        if (fabs(C_ref[i] - C_test[i]) > tolerance)
        {
            printf("Mismatch at index %d: ref=%f, test=%f\n", i, C_ref[i], C_test[i]);
            return 0;
        }
    }
    return 1;
}

typedef void (*matmul_fn)(const float *A, const float *B, float *C, int M, int N, int K);

float benchmark(matmul_fn matmul, const float *A, const float *B, float *C, int M, int N, int K)
{
    matmul(A, B, C, M, N, K); // Warmup
    double total_time = 0.0;
    for (int i = 0; i < num_iterations; i++)
    {
        double start = get_time_ms();
        matmul(A, B, C, M, N, K);
        __asm__ __volatile__("" : "+m"(C[0]) : : "memory");
        double end = get_time_ms();
        total_time += end - start;
    }

    return total_time / num_iterations;
}

// ============================================================================
// Main: Verify results and performance benchmakrk
// ============================================================================
int main(int argc, char *argv[])
{
    srand(42);
    printf("MatMul Benchmark: Square Matrix\n");

    int sizes[] = {1024}; //{512, 256, 128, 64};
    int n = sizeof(sizes) / sizeof(sizes[0]);

    printf("%-8s %-15s %-15s %-15s %-15s\n", "Size", "Naive", "Reordered", "Tiled", "Parallel");
    printf("%-8s %-15s %-15s %-15s %-15s\n", "----", "-----", "---------", "-----", "--------");

    for (int i = 0; i < n; i++)
    {
        int M = sizes[i], N = M, K = M;

        float *A = (float *)malloc(M * K * sizeof(float));
        float *B = (float *)malloc(K * N * sizeof(float));
        float *C = (float *)malloc(M * N * sizeof(float));
        float *C_ref = (float *)malloc(M * N * sizeof(float));

        initialize_matrix(A, M, K);
        initialize_matrix(B, K, N);

        // --- 1. Naive ---
        memset(C, 0, M * N * sizeof(float));

        float t_naive = benchmark(matmul_naive, A, B, C_ref, M, N, K);
        double g_naive = calculate_gflops(M, N, K, t_naive);

        // --- 2. Tiled ---
        memset(C, 0, M * N * sizeof(float));

        float t_blocking = benchmark(matmul_looptiling, A, B, C, M, N, K);
        double g_blocking = calculate_gflops(M, N, K, t_blocking);
        if (!verify_result(C_ref, C, M, N, 1e-3))
        {
            printf("Tiled implementation failed verification!\n");
            return 1;
        }

        // --- 3. Reordered ---
        memset(C, 0, M * N * sizeof(float));

        float t_reorder = benchmark(matmul_looporder, A, B, C, M, N, K);
        double g_reorder = calculate_gflops(M, N, K, t_reorder);
        if (!verify_result(C_ref, C, M, N, 1e-3))
        {
            printf("Reordered implementation failed verification!\n");
            return 1;
        }

        // --- 4. Parallel ---
        memset(C, 0, M * N * sizeof(float));

        float t_parallel = benchmark(matmul_parallel_ikj, A, B, C, M, N, K);
        double g_parallel = calculate_gflops(M, N, K, t_parallel);
        if (!verify_result(C_ref, C, M, N, 1e-3))
        {
            printf("Parallel implementation failed verification!\n");
            return 1;
        }
        printf("%d\t%.2f GFLOPS\t%.2f GFLOPS\t%.2f GFLOPS\t%.2f GFLOPS\n",
               M, g_naive, g_reorder, g_blocking, g_parallel);

        free(A);
        free(B);
        free(C);
    }

    return 0;
}