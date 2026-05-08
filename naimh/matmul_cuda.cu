/*
 * Praktikum 1 – Matrix Multiplication on GPU (CUDA)
 * ===================================================
 * AI Accelerators (AIA) – Task 8 / Extra
 *
 * Three implementations:
 *   1. Naive CUDA kernel     – one thread per output element
 *   2. Tiled shared-memory   – TILE×TILE thread blocks, data staged in SMEM
 *   3. cuBLAS SGEMM          – library baseline (GPU equivalent of MKL)
 *
 * Build (after installing CUDA Toolkit):
 *   nvcc -O3 -arch=sm_89 matmul_cuda.cu -lcublas -o matmul_cuda
 *
 * Run:
 *   ./matmul_cuda
 *
 * sm_89 = Ada Lovelace (RTX 4070 Super). Adjust for other GPUs:
 *   sm_86 = Ampere (RTX 30xx), sm_75 = Turing (RTX 20xx)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TILE 32
#define NUM_ITERS 10
#define TOLERANCE 1e-2f

/* ── Error-checking macros ─────────────────────────────────────────────── */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t s = (call);                                              \
        if (s != CUBLAS_STATUS_SUCCESS) {                                       \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n",                        \
                    __FILE__, __LINE__, (int)s);                                \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

/* ── GPU info ───────────────────────────────────────────────────────────── */
static void print_device_info(void) {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("GPU: %s\n", p.name);
    printf("  SMs: %d   CUDA cores/SM: %d   Boost: %.0f MHz\n",
           p.multiProcessorCount, p.maxThreadsPerMultiProcessor,
           (double)p.clockRate / 1000.0);
    printf("  Global mem: %.0f MB   Bandwidth: ~%.0f GB/s\n",
           (double)p.totalGlobalMem / (1024 * 1024),
           2.0 * p.memoryClockRate * 1e3 * (p.memoryBusWidth / 8) / 1e9);
    printf("  Shared mem/block: %zu KB   L2: %d MB\n\n",
           p.sharedMemPerBlock / 1024, p.l2CacheSize / (1024 * 1024));
}

/* ── Utility ────────────────────────────────────────────────────────────── */
static void init_matrix(float *m, int n) {
    for (int i = 0; i < n * n; i++)
        m[i] = (float)(rand() % 100);
}

static int verify(const float *ref, const float *test, int n) {
    for (int i = 0; i < n * n; i++) {
        float scale = fmaxf(fabsf(ref[i]), 1.0f);
        if (fabsf(ref[i] - test[i]) > TOLERANCE * scale) {
            printf("  MISMATCH @ %d: ref=%.4f  test=%.4f\n", i, ref[i], test[i]);
            return 0;
        }
    }
    return 1;
}

static double gflops(int n, double ms) {
    return (2.0 * n * n * n) / (ms / 1000.0) / 1e9;
}

/* ── Timing via CUDA events ─────────────────────────────────────────────── */
typedef void (*kernel_launcher)(const float *, const float *, float *, int);

static double benchmark_kernel(kernel_launcher fn,
                                const float *d_A, const float *d_B, float *d_C,
                                int n) {
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    fn(d_A, d_B, d_C, n);  /* warmup */
    CUDA_CHECK(cudaDeviceSynchronize());

    float total_ms = 0.0f;
    for (int i = 0; i < NUM_ITERS; i++) {
        CUDA_CHECK(cudaEventRecord(t0));
        fn(d_A, d_B, d_C, n);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        total_ms += ms;
    }

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return total_ms / NUM_ITERS;
}

/* ════════════════════════════════════════════════════════════════════════ */
/*  KERNEL 1 – NAIVE                                                        */
/*  One thread computes one element of C.                                   */
/*  C[i,j] = sum_k A[i,k] * B[k,j]                                         */
/*  B is accessed column-wise → uncoalesced reads → slow at large N.       */
/* ════════════════════════════════════════════════════════════════════════ */
__global__ void kernel_naive(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < N; k++)
        sum += A[row * N + k] * B[k * N + col];
    C[row * N + col] = sum;
}

static void launch_naive(const float *A, const float *B, float *C, int n) {
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    kernel_naive<<<grid, block>>>(A, B, C, n);
}

/* ════════════════════════════════════════════════════════════════════════ */
/*  KERNEL 2 – TILED SHARED MEMORY                                          */
/*  Each TILE×TILE thread block accumulates a TILE×TILE output tile.        */
/*  A-tile and B-tile are loaded into shared memory first, then reused      */
/*  by all threads in the block → coalesced loads, ~TILE× bandwidth saving. */
/* ════════════════════════════════════════════════════════════════════════ */
__global__ void kernel_tiled(const float *A, const float *B, float *C, int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
        /* Load one tile of A and B into shared memory (coalesced) */
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        /* Each thread accumulates its dot-product slice over this tile */
        #pragma unroll
        for (int k = 0; k < TILE; k++)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = acc;
}

static void launch_tiled(const float *A, const float *B, float *C, int n) {
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    kernel_tiled<<<grid, block>>>(A, B, C, n);
}

/* ════════════════════════════════════════════════════════════════════════ */
/*  KERNEL 3 – cuBLAS SGEMM                                                 */
/*  Library baseline: hand-tuned tensor-core microkernel, equivalent to     */
/*  MKL on the CPU side. Uses column-major convention; we transpose A and B  */
/*  via the transa/transb flags instead of actually transposing memory.      */
/* ════════════════════════════════════════════════════════════════════════ */
static cublasHandle_t cublas_handle;

static void launch_cublas(const float *A, const float *B, float *C, int n) {
    const float alpha = 1.0f, beta = 0.0f;
    /* cuBLAS is column-major. To compute C = A*B in row-major, we use:
     *   C^T = B^T * A^T
     * which in cuBLAS terms is: C = op(B) * op(A) with both ops = NOTRANS,
     * passing B as "A" and A as "B". */
    CUBLAS_CHECK(cublasSgemm(cublas_handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             n, n, n,
                             &alpha,
                             B, n,
                             A, n,
                             &beta,
                             C, n));
}

/* ════════════════════════════════════════════════════════════════════════ */
/*  Main                                                                     */
/* ════════════════════════════════════════════════════════════════════════ */
int main(void) {
    srand(42);
    print_device_info();
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    int sizes[] = {256, 1024, 4096};
    int n_sizes = (int)(sizeof(sizes) / sizeof(sizes[0]));

    printf("%-8s %-18s %-18s %-18s\n",
           "N", "Naive (GFLOP/s)", "Tiled (GFLOP/s)", "cuBLAS (GFLOP/s)");
    printf("%-8s %-18s %-18s %-18s\n",
           "----", "---------------", "---------------", "----------------");

    for (int s = 0; s < n_sizes; s++) {
        int n = sizes[s];
        size_t bytes = (size_t)n * n * sizeof(float);

        /* Host matrices */
        float *h_A = (float *)malloc(bytes);
        float *h_B = (float *)malloc(bytes);
        float *h_C_ref  = (float *)malloc(bytes);
        float *h_C_test = (float *)malloc(bytes);
        init_matrix(h_A, n);
        init_matrix(h_B, n);

        /* Device matrices */
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

        /* Verify tiled against naive (skip at N=4096 — too slow for naive) */
        if (n <= 1024) {
            CUDA_CHECK(cudaMemset(d_C, 0, bytes));
            launch_naive(d_A, d_B, d_C, n);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_C_ref, d_C, bytes, cudaMemcpyDeviceToHost));

            CUDA_CHECK(cudaMemset(d_C, 0, bytes));
            launch_tiled(d_A, d_B, d_C, n);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_C_test, d_C, bytes, cudaMemcpyDeviceToHost));

            if (!verify(h_C_ref, h_C_test, n))
                printf("  [!] Tiled kernel verification FAILED at N=%d\n", n);
        }

        /* Benchmark */
        double ms_naive  = benchmark_kernel(launch_naive,  d_A, d_B, d_C, n);
        double ms_tiled  = benchmark_kernel(launch_tiled,  d_A, d_B, d_C, n);
        double ms_cublas = benchmark_kernel(launch_cublas, d_A, d_B, d_C, n);

        printf("%-8d %-18.2f %-18.2f %-18.2f\n",
               n, gflops(n, ms_naive), gflops(n, ms_tiled), gflops(n, ms_cublas));

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        free(h_A); free(h_B); free(h_C_ref); free(h_C_test);
    }

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    return 0;
}
