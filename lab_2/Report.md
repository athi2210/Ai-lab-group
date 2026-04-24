# Lab Report – Matrix Multiplication on CPU
**Course:** AI Accelerators (AIA)
**Lab:** Praktikum 1
**Team members:** _(fill in)_
**Date:** _(fill in)_

---

## Task 1 – System Characterisation

> Fill in the details of your machine. Use tools such as `lscpu`, `lstopo`, `/proc/cpuinfo`.

| Property | Value |
|---|---|
| CPU model | Apple M2 |
| Number of cores / threads | 8/8 |
| Base / Boost clock speed (GHz) | 2.42/­3.5 |
| SIMD ISA (SSE4.2 / AVX2 / AVX-512 …) | |
| SIMD width (bits / floats per vector) | |
| MAC units per core | |
| L1 cache size (per core) | hw.l1icachesize: 131072 |
| L2 cache size (per core) | hw.l1dcachesize: 65536 |
| L3 cache size (shared) | hw.l2cachesize: 4194304 |
| Peak theoretical throughput (GFLOP/s) | |

**How did you calculate peak throughput?**

_(formula: cores × clock × SIMD_width × MACs_per_cycle)_

---

## Task 2 – Loop Reordering

> Measure each loop ordering for matrix sizes 64, 128, 256, 512, 1024, 2048, 4096.

| Loop order | N=512 (GFLOP/s) | N=256 (GFLOP/s) | N=128 (GFLOP/s) | N=64 (GFLOP/s) |
|---|---|---|---|---|
|i-j-k| 1.99 | 2.30 | 2.94 | 3.95 |
|i-k-j| 28.25 | 27.92 | 33.42 | 25.27 |
|j-i-k| 1.86 | 2.03 | 2.51 | 3.62 |
|j-k-i| 0.54 | 0.62 | 2.31 | 2.43 |
|k-i-j| 22.31 | 22.80 | 33.03 | 30.39 |
|k-j-i| 0.54 | 0.64 | 3.25 | 3.91 |

**Best ordering found:** ikj

**Why does this ordering perform best?**

_(Explain in terms of spatial locality and cache reuse of A, B, and C)_
A[i][k] is reused and 
B[k][j] is used and then the next entry B[k][j+1]
It also uses the immediate next entry for C


---

## Task 3 – Vectorization

> List the compiler flags you tested and their effect.

| Flags added | N=1024 (GFLOP/s) | Speedup vs. naive |
|---|---|---|
| -O3 only (baseline) | 1.83| 1.0× |
| -O3 -march=native | 1.82 | 1.0x|
| -O3 -march=native -ffast-math | 1.80 | |
| -O3 -march=native -ffast-math -funroll-loops | 1.84| |
| -O3 -march=native -ffast-math -fopenmp-simd | 1.84| |

**Did you add any `#pragma` hints to the source?** If yes, which ones?

**What speedup did you achieve? Why?**

---

## Task 4 – Loop Tiling

> Experiment with tile sizes to find the sweet spot for your cache hierarchy.

| Tile size | N=1024 (GFLOP/s) | N=4096 (GFLOP/s) |
|---|---|---|
| 32 | | |
| 64 | | |
| 128 | | |
| 256 | | |

**Best tile size:** ___

**Why does this tile size work best for your machine?**

---

## Task 5 – Multithreading

> Measure scaling as you increase the number of OpenMP threads.

| Threads | N=4096 (GFLOP/s) | Speedup |
|---|---|---|
| 1 | | 1.0× |
| 2 | | |
| 4 | | |
| 8 | | |
| _(max physical cores)_ | | |

**Does throughput scale linearly with threads?** Why / why not?

---

## Task 6 – Performance Analysis

**Is your implementation compute-bound or memory-bound?** Justify with arithmetic intensity (FLOPs / bytes).

**Comparison vs. PyTorch (N=4096):**

| Implementation | GFLOP/s | % of PyTorch |
|---|---|---|
| Naive C | | |
| Best optimised C | | |
| PyTorch (CPU) | | 100% |

**What is the gap and why does it exist?**

---

## Task 7 – Key Takeaways

_Write 3–5 sentences summarising the most important lessons learned from this lab._

---

## Figures

> Place your performance plots (GFLOP/s vs. matrix size) in the `figures/` folder and reference them here.

![Performance comparison](figures/performance.png)
