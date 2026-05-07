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

#define num_threads 4
const int num_iterations = 4;
#define JB 64 //Tile size divides matrix size

// ============================================================================
// IMPLEMENTATION 1: NAIVE MATRIX MULTIPLICATION
// ============================================================================
void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// IMPLEMENTATION 2: -03 -ffmastmath, does loop unrolling and vectorization, reorder k,j
// ============================================================================
void matmul_looporder(const float* A, const float* B, float* C, int M, int N, int K) {
    ikj
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            //#pragma GCC ivdep
            for (int j = 0; j < N; j++)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
        }

    // // jki
    // for (int j = 0; j < N; j++)
    //     for (int k = 0; k < K; k++)
    //         for (int i = 0; i < M; i++)
    //             C[i * N + j] += A[i * K + k] * B[k * N + j];

    // // kij
    // for (int k = 0; k < K; k++)
    //     for (int i = 0; i < M; i++)
    //         for (int j = 0; j < N; j++)
    //             C[i * N + j] += A[i * K + k] * B[k * N + j];
   
    
}

// ============================================================================
// IMPLEMENTATION 3: Tiling
// ============================================================================

void matmul_looptiling(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i += JB)
        for (int k = 0; k < K; k += JB)
            for (int j = 0; j < N; j += JB)
                for (int ii = i; ii < i + JB && ii < M; ii++)
                    for (int kk = k; kk < k + JB && kk < K; kk++) {
                        for (int jj = j; jj < j + JB && jj < N; jj++)
                            C[ii * N + jj] += A[ii * K + kk] * B[kk * N + jj];
                    }

}

// ============================================================================
// IMPLEMENTATION 4: Multithreading
// ============================================================================
void matmul_parallel_ikj(const float* A, const float* B, float* C,
                         int M, int N, int K) {
    /* TODO: combine a tiling with thread parallelism */
    //  #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < M; i += JB)
        for (int k = 0; k < K; k += JB)
            for (int j = 0; j < N; j += JB)
         #pragma omp parallel for num_threads(num_threads)
                for (int ii = i; ii < i+JB && ii < M; ii++)
                    for (int kk = k; kk < k+JB && kk < K; kk++)
                        for (int jj = j; jj < j+JB && jj < N; jj++)
                            C[ii*N+jj] += A[ii*K+kk] * B[kk*N+jj];
    matmul_naive (A, B, C, M, N, K);
}

// ============================================================================
// Utility functions: Init Matrix, Benchmarking, Calculate Gflops
// ============================================================================
void initialize_matrix(float *matrix, int rows, int cols){
    for (int i = 0; i < rows * cols; i++){
        matrix[i] = rand() % 100;
    }
}

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

double calculate_gflops(int M, int N, int K, double total_time) {
    double flops = 2.0 * M * N * K;
    double gflops = (flops / ((total_time) / 1000.0)) / 1e9;
    return gflops;
}

int verify_result(const float* C_ref, const float* C_test, int M, int N, float tolerance) {
    for (int i = 0; i < M * N; i++) {
        if (fabs(C_ref[i] - C_test[i]) > tolerance) {
            printf("Mismatch at index %d: ref=%f, test=%f\n", i, C_ref[i], C_test[i]);
            return 0;
        }
    }
    return 1;
}

typedef void (*matmul_fn)(const float* A, const float* B, float* C, int M, int N, int K);

float benchmark(matmul_fn matmul, const float* A, const float *B, float *C, int M, int N, int K)
{
    matmul(A, B, C, M, N, K); //Warmup
    double total_time = 0.0;
    for (int i = 0; i < num_iterations; i++) {
        double start = get_time_ms();
        matmul(A, B, C, M, N, K);
        __asm__ __volatile__("" : "+m" (C[0]) : : "memory");
        double end = get_time_ms();
        total_time += end - start;
    }

    return total_time/num_iterations;
}

// ============================================================================
// Main: Verify results and performance benchmakrk
// ============================================================================
int main(int argc, char *argv[]) {
    srand(42);
    printf("MatMul Benchmark: Square Matrix\n");

    int sizes[] = {1024, 256, 128, 64}; // Different sizes to test
    int n = sizeof(sizes) / sizeof(sizes[0]);// number of different sizes to test

    printf("%-8s %-15s %-15s %-15s %-15s\n", "Size", "Naive", "Reordered", "Tiled", "Parallel");
    printf("%-8s %-15s %-15s %-15s %-15s\n", "----", "-----", "---------", "-----", "--------");

    for (int i = 0; i < n; i++) {
        int M = sizes[i], N = M, K = M;

        float *A    = (float *)malloc(M * K * sizeof(float));
        float *B    = (float *)malloc(K * N * sizeof(float));
        float *C    = (float *)malloc(M * N * sizeof(float));
        float *C_ref = (float *)malloc(M * N * sizeof(float));

        initialize_matrix(A, M, K);
        initialize_matrix(B, K, N);

        // --- 1. Naive (reference) ---
        memset(C_ref, 0, M * N * sizeof(float));
        float t_naive = benchmark(matmul_naive, A, B, C_ref, M, N, K);
        double g_naive = calculate_gflops(M, N, K, t_naive);

        // --- 2. Reordered ---
        memset(C, 0, M * N * sizeof(float));
        float t_reorder = benchmark(matmul_looporder, A, B, C, M, N, K);
        double g_reorder = calculate_gflops(M, N, K, t_reorder);
        if (!verify_result(C_ref, C, M, N, 1e-1)) printf("  [FAIL] Reordered N=%d\n", M);

        // --- 3. Tiled ---
        memset(C, 0, M * N * sizeof(float));
        float t_blocking = benchmark(matmul_looptiling, A, B, C, M, N, K);
        double g_blocking = calculate_gflops(M, N, K, t_blocking);
        if (!verify_result(C_ref, C, M, N, 1e-1)) printf("  [FAIL] Tiled N=%d\n", M);

        // --- 4. Parallel ---
        memset(C, 0, M * N * sizeof(float));
        float t_parallel = benchmark(matmul_parallel_ikj, A, B, C, M, N, K);
        double g_parallel = calculate_gflops(M, N, K, t_parallel);
        if (!verify_result(C_ref, C, M, N, 1e-1)) printf("  [FAIL] Parallel N=%d\n", M);

        printf("%d\t%.2f GFLOPS\t%.2f GFLOPS\t%.2f GFLOPS\t%.2f GFLOPS\n",
               M, g_naive, g_reorder, g_blocking, g_parallel);

        free(A); free(B); free(C); free(C_ref);
    }

    return 0;
}