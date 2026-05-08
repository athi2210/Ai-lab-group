# Lab Report – Matrix Multiplication on CPU
**Course:** AI Accelerators (AIA)
**Lab:** Praktikum 1
**Team members:** _(fill in)_
**Date:** _(fill in)_

---

## Task 1 – System Characterisation

> Fill in machine details using `lscpu`, `lstopo`, `/proc/cpuinfo`. Calculate theoretical peak GFLOP/s.

| Property | Value |
|---|---|
| CPU model | AMD Ryzen 7 7800X3D (Zen 4) |
| Cores / threads | 8 / 16 (8 physical + SMT) |
| Base / Boost clock | 4.2 / 5.0 GHz |
| SIMD ISA | AVX-512 (full ISA support) |
| SIMD width | 512 bits / 16 FP32 lanes (architectural) — 256 bits physical (double-pumped) |
| FMA / MAC units per core | 2 × 256-bit |
| L1 cache (per core) | 32 KB I + 32 KB D = 64 KB total |
| L2 cache (per core) | 1 MB |
| L3 cache (shared) | 96 MB (32 MB + 64 MB stacked 3D V-Cache) |
| Memory | 32 GB DDR5-5600 (running 5200 MT/s), ~80 GB/s peak bandwidth |

**Peak throughput calculation:**

```
ISA-based:    8 cores × 5.0 GHz × 16 FP32 lanes × 2 FMA pipes × 2 (FMA = mul+add) = 2560 GFLOP/s
Effective:    8 cores × 5.0 GHz ×  8 FP32 lanes × 2 FMA pipes × 2                 = 1280 GFLOP/s
At base clock (4.2 GHz, all-core sustained):                                        1075 GFLOP/s
```

For Tasks 4–6 we use **1075 GFLOP/s** as the realistic ceiling.

### Difference ISA and effective

The Zen 4 Architecture supports the full AVX-512 instruction set. The FMA units, however, are physically only 256 bit wide. Each AVX-512 instruction is split into two 256 bit operations and double-pumped in two cycles.

Effectively we only have a 256 bit SIMD width but we are allowed AVX-512 instructions.

---

## Task 2 – Loop Reordering

> Implement the best loop ordering. Try i-k-j, j-k-i, k-i-j, and others. Record results.

GFLOP/s with multiplier vs. i-j-k naive baseline at the same N in parens.

| Loop order | N=256 | N=1024 | N=4096 |
|---|---|---|---|
| i-j-k (naive) | 3.20 (1.0×) | 0.83 (1.0×) | 0.35 (1.0×) |
| **i-k-j** | **16.99 (5.3×)** | **22.36 (26.9×)** | **16.04 (45.8×)** |
| k-i-j | 14.41 (4.5×) | 18.96 (22.8×) | 17.67 (50.5×) |
| j-i-k | 3.07 (0.96×) | 0.59 (0.71×) | 0.37 (1.06×) |
| j-k-i | 0.90 (0.28×) | 0.10 (0.12×) | ~0.04* (~0.11×) |
| k-j-i | 0.70 (0.22×) | 0.11 (0.13×) | ~0.05* (~0.14×) |

*Estimated from N=1024 trend × the 0.42 naive scaling factor going N=1024 → N=4096. Not measured directly — at this throughput a single call takes ~50-60 minutes due to catastrophic cache thrashing (every C and A access strides by N×4 bytes = 16 KB at N=4096, exceeding cache line size by 256×).

**Best ordering: i-k-j** (27× faster than naive at N=1024).
---

## Task 3 – Vectorization

> List the compiler flags you tested and their effect.

Built the same source 5 times, measured **both** naive AND i-k-j in each build to isolate flag effects from algorithm effects.

| Flags added | Naive GFLOP/s | Naive speedup | i-k-j GFLOP/s | i-k-j speedup |
|---|---|---|---|---|
| -O3 only (baseline) | 0.83 | 1.0× | 22.98 | 1.0× |
| -O3 -march=native | 0.81 | 0.98× | 31.04 | 1.35× |
| **-O3 -march=native -ffast-math** | 0.74 | 0.89× | **36.37** | **1.58×** |
| ... -funroll-loops | 0.73 | 0.88× | 34.56 | 1.50× |
| ... -fopenmp-simd | 0.71 | 0.86× | 36.24 | 1.58× |

**Best flag set: `-O3 -march=native -ffast-math`** (1.58× over -O3 alone on i-k-j).

The naive column trends *downward* with more flags — wider AVX-512 vectorization makes B's stride-N gathers more expensive, not less. Flags only help when the algorithm provides stride-1 streams.

---

## Task 4 – Loop Tiling

> Experiment with tile sizes to find the sweet spot for your cache hierarchy.

Tiled i-k-j ordering, fast flags, single-level tiling. Both `malloc` and `aligned_alloc(64)` columns shown.

| Tile size | N=1024 `malloc` | N=1024 `aligned_alloc(64)` | N=4096 `malloc` | N=4096 `aligned_alloc(64)` | Working set |
|---|---|---|---|---|---|
| 32 | 8.43 | 12.24 (+45%) | 7.60 | 10.72 (+41%) | 12 KB |
| 64 | 14.73 | 29.36 (+99%) | 16.28 | 24.57 (+51%) | 48 KB |
| 128 | 21.51 | 45.63 (+112%) | 31.66 | 50.59 (+60%) | 192 KB |
| **256** | 34.08 | **48.22 (+41%)** | 44.69 | **60.75 (+36%)** | 768 KB |
| 512 | 32.03 | 38.35 (+20%) | 34.17 | 44.78 (+31%) | 3 MB |

**Best tile size: T = 256.**

#### Two-level (cubic) tiling

| Strategy | Configuration | N=4096 GFLOP/s | vs T=256 single-level |
|---|---|---|---|
| **Single-level** | **T=256** | **44.69** | **1.00×** |
| 2-level | T2=1024, T1=128 | 21.34 | 0.48× |
| 2-level | T2=512, T1=128 | 21.37 | 0.48× |
| 2-level | T2=256, T1=128 | 20.51 | 0.46× |
| 2-level | T2=512, T1=64 | 13.65 | 0.31× |

#### Memory alignment

Switching from `malloc` to `aligned_alloc(64, ...)` gave +25% to +112% speedups depending on tile size. Smaller tiles benefit more (more memory ops per FLOP).

---

## Task 5 – Multithreading

> Measure scaling as you increase the number of OpenMP threads.

Tiled i-k-j (T=256, fast flags, aligned), single `#pragma omp parallel for schedule(static)` on outer `ii` loop. Pinning: `OMP_PROC_BIND=spread OMP_PLACES=cores`.

| Threads | N=4096 GFLOP/s | Speedup | Efficiency |
|---|---|---|---|
| 1 | 56.02 | 1.00× | 100% |
| 2 | 106.04 | 1.89× | 95% |
| 4 | 195.14 | 3.48× | 87% |
| 8 (max physical) | 304.03 | 5.43× | 68% |
| 16 (SMT enabled) | **388.65** | **6.94×** | 43% |

Efficiency = actual speedup / N. 100% = perfect linear scaling.

---

## Task 6 – Performance Analysis

> Is your implementation compute-bound or memory-bound? Justify with arithmetic intensity. Compare vs PyTorch (N=4096). Explain the gap.

### Compute-bound or memory-bound?

The answer depends on the kernel. The roofline crossover for this machine is:

```
Peak compute  =  1075 GFLOP/s
Peak bandwidth =    80 GB/s
Crossover AI  =  1075 / 80  =  ~13.4 FLOPs/byte
```

Above 13.4 FLOPs/byte → compute-bound. Below → memory-bound.

| Kernel | What gets reloaded | Approx AI | Bound by |
|---|---|---|---|
| Naive (i-j-k) | Whole B reloaded N times | ~0.5 | Latency-bound (below DRAM streaming ceiling) |
| i-k-j + flags | Cache-friendly, but no tile reuse | ~5–10 | L2/L3 bandwidth |
| **Tiled T=256, threaded** | Each tile reused N/T times | **~43** | **Compute-bound** |
| PyTorch / MKL | Same + register-tiled C | ~43 | Compute-bound, near peak |

The naive kernel sits at ~0.5 FLOPs/byte — below even the DRAM streaming ceiling of 0.5 × 80 = 40 GFLOP/s. Measured: 0.35 GFLOP/s. The strided column access into B causes cache misses that stall for ~200 cycles each; the prefetcher cannot predict stride-N accesses, so it is effectively latency-bound.

The tiled+threaded kernel reaches 388.65 GFLOP/s = **36% of the 1075 GFLOP/s ceiling**. With AI ≈ 43 it is firmly compute-bound; the remaining 64% gap to peak is FMA pipe utilisation, not memory.

### Comparison vs PyTorch (N=4096, FP32)

| Implementation | Threads | GFLOP/s | % of peak | × over naive |
|---|---|---|---|---|
| Naive C (i-j-k, -O3) | 1 | 0.35 | 0.03% | 1× |
| **Best optimised C** | 16 | **388.65** | **36%** | **1110×** |
| PyTorch (MKL) | 1 | 123.06 | 11% | 352× |
| PyTorch (MKL) | 8 | 651.97 | 61% | 1863× |
| **PyTorch (MKL)** | 16 | **887.12** | **82%** | **2535×** |

Our best achieves **44% of PyTorch** at 16 threads. PyTorch is **2.3× faster** on identical arithmetic.

### Why the gap exists

| Technique | What it does | Why we can't match it |
|---|---|---|
| **Register tiling** | Keeps a 4×16 tile of C in vector registers for the entire k-loop — C never touches L1 during FMAs | Requires hand-written assembly; compiler cannot guarantee register residency |
| **Software pipelining** | Issues many independent FMAs ahead to hide the 4-cycle FMA latency | Compiler register allocation is too conservative at scale |
| **Multi-level tiling** | Outer→L3, middle→L2, inner→L1, microtile→registers | Our single-level tiling only targets L2 |
| **Specialised microkernels** | Different code paths per matrix shape and CPU | We use one generic loop |

The gap is structural. Our cubic tiling reaches 36% of peak; production BLAS reaches 80%+ through register-level scheduling that no compiler generates automatically.

---

## Cumulative Journey at N=4096

Each stage is one experiment, the finding it produced, and what carried into the final [matmul.c](matmul.c). Numbers cross-reference the per-task tables above.

| # | Stage | Experiment (source) | Key finding | GFLOP/s | × naive | What carried into matmul.c |
|---|---|---|---|---|---|---|
| 0 | **Naive baseline** | — | i-j-k, `-O3` only — strided B reads, no SIMD | 0.35 | 1× | `matmul_naive()` — reference floor |
| 1 | **Loop ordering** | Task 2 — all 6 orderings tested | j-innermost gives stride-1 on B & C → vectorizable. **i-k-j wins**; k-j-i is 200× slower on identical math | 16.04 | 46× | `matmul_looporder()` i-k-j; same inner nest reused in every later kernel |
| 2 | **Compiler flags** | Task 3 — flag sweep on i-k-j | `-O3 -march=native -ffast-math` enables AVX-512 FMA + reduction reassociation; **+1.65×** at N=4096 (16.04→26.42) | 26.42 | 75× | Makefile flags |
| 3 | **Single-level tiling** | Task 4 — sweep T ∈ {32, 64, 128, 256, 512} | U-shape: T must allow long AVX inner loop AND fit in L2. **T=256 wins**; T=512 regresses 23% from L2 spill | 44.69 | 128× | `#define TILE 256` + `matmul_looptiling()` |
| 4 | **2-level tiling** | Task 4 — T1×T2 grid (negative result) | Cubic 2-level can't satisfy *long inner loop* AND *L1-resident* simultaneously. Best config T2=512/T1=128 → 21.37, ~2× slower than single-level T=256 | 21.37 | 61× | **Nothing** — single-level kept |
| 5 | **Memory alignment** | Task 4 — `malloc` vs `aligned_alloc(64)` | AVX-512 prefers cache-line-aligned data; misalignment causes line-crossing penalty. **+25–112%** depending on T | 60.75 | 174× | `aligned_alloc(64, …)` for A, B, C |
| 6 | **Threading** | Task 5 — sweep N ∈ {1,2,4,8} threads × `OMP_PROC_BIND` × `OMP_PLACES` | 8 cores @ 87→68% efficiency (DDR5 + all-core boost throttle); `BIND=spread`, `PLACES=cores` to avoid SMT-sibling collisions | 304.03 | 869× | `#pragma omp parallel for schedule(static)` on outer `ii` |
| 7 | **SMT** | Task 5 — 16 threads vs 8 | **Surprise: +28%**. Kernel isn't 100% FMA-saturated → second SMT sibling fills loop overhead, address arithmetic, memset gaps | **388.65** | **1110×** | Runtime knob: `OMP_NUM_THREADS=16` (not source) |

**Total: 1110× speedup, identical math, no GPU, no hand-written assembly.**

The 56% gap remaining to PyTorch is the boundary between "good engineering" and "decades of BLAS register-level tuning" — see Task 6's *Why the gap exists*. Task 8 below shifts to a different architecture entirely.

---

## Task 7 – Key Takeaways

> 3–5 sentences summarising the most important lessons learned.

1. **Access patterns matter** Doing the same mathematical operation in a different pattern increases the speed of which i t can be done. BEing able to vectorize is important.

2. **Filling up CPU registries correctly matters** Trying not to waste expensive memory writes seems like the most important thing (both tile size and aligned_alloc).

3. **CPU flags dont add speed, they multiply it** Accessing the matricies "incorrectly" and then trying to speed up the math actually slows it down. Only with a solid baseline do we get a good result.

---

## Task 8 (Extra) – GPU Acceleration with CUDA

> Extend the lab to GPU. Implement naive, tiled (shared-memory), and cuBLAS variants in FP32, TF32, FP16, and BF16. Add a structured 2:4 sparse FP16 path via cuSPARSELt. Sweep N up to 16384. Compare against the CPU best.

### Hardware

| Property | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 SUPER (Ada Lovelace, AD104) |
| SMs / FP32 cores | 56 / 7168 (128 per SM) |
| Tensor cores | 224 (4 per SM, 4th-gen) |
| Boost clock | 2.55 GHz (measured via `cudaDeviceGetAttribute`) |
| Global memory | 12 GB GDDR6X |
| Memory bandwidth | ~504 GB/s |
| L2 cache | 48 MB |
| Shared memory / block | 48 KB (used here; up to 100 KB available with opt-in) |

**Peak throughput (NVIDIA spec, all on the same silicon — different precision paths):**

```
FP32 (CUDA cores):          56 × 128 × 2 (FMA) × 2.55 GHz = 36,556 GFLOP/s
TF32 (tensor cores):        ~73,000 GFLOP/s    (≈ 2× FP32 peak)
FP16/BF16 (tensor cores):   ~142,000 GFLOP/s   (≈ 4× FP32 peak)
FP16 + 2:4 sparsity:        ~284,000 GFLOP/s   (≈ 8× FP32 peak — not measured)
```

### Six implementations

1. **Naive** — one thread per output element. B read column-wise → uncoalesced.
2. **Tiled (shared-memory)** — 32×32 thread-block tile, A-tile + B-tile staged in SMEM.
3. **cuBLAS SGEMM (FP32)** — vendor library on CUDA cores. GPU equivalent of MKL.
4. **cuBLAS SGEMM (TF32)** — same call, `cublasSetMathMode(CUBLAS_TF32_TENSOR_OP_MATH)`. FP32 layout in/out, TF32 multiply on tensor cores (10-bit mantissa, 8-bit exponent).
5. **cuBLAS GEMM-Ex (FP16)** — `cublasGemmEx` with `CUDA_R_16F` inputs, `CUBLAS_COMPUTE_32F` accumulator. FP16 layout (5-bit exp, 10-bit mantissa).
6. **cuBLAS GEMM-Ex (BF16)** — same as FP16 but `CUDA_R_16BF`. Same dynamic range as FP32, 7-bit mantissa.

Built with `nvcc -O3 -arch=sm_89 matmul_cuda.cu -lcublas`. Random FP32 input is converted to FP16/BF16 once via a one-shot device kernel before benchmarking.

### Results

| N | Naive | Tiled | FP32 | TF32 | FP16 | BF16 |
|---|---|---|---|---|---|---|
| 256   | 1114 | 1366 | 3921  | 4472  | 4179  | 3575  |
| 1024  | 1977 | 2713 | 14791 | 24750 | 41341 | 38548 |
| 4096  | 2033 | 2685 | 27509 | 35000 | 73215 | 71965 |
| 8192  | —    | —    | 25263 | 38631 | 76533 | 76495 |
| 16384 | —    | —    | 23691 | 39324 | **77381** | 77106 |

(Naive and tiled skipped at N≥8192 — would take ~30 s each per benchmark.)

**New ceiling: 77.4 TFLOPS at FP16 / N=16384**, vs 388.65 GFLOP/s on CPU best = **199× speedup over the entire CPU optimisation arc**.

### What the precision sweep reveals

#### 1. FP16 ≈ BF16 — same hardware path

At every N ≥ 1024, FP16 and BF16 are within 5% of each other and plateau at ~77 TFLOPS. The 16-bit format choice is a numerical-precision decision (BF16's wider exponent vs FP16's wider mantissa), not a performance one — the same tensor-core silicon executes both.

#### 2. FP16/BF16 plateau at 54% of theoretical peak

Once N ≥ 4096, the 16-bit kernels saturate at ~77 TFLOPS regardless of size. **77 / 142 = 54% of FP16 tensor-core peak.** The remaining 46% headroom is register-file pressure and warp-tile-emit cadence — the same kind of "library leaves room for hand-tuned kernels" gap we've seen at every level. Closing it would need CUTLASS-style hand-coded warp tiles using the `mma.sync` PTX instruction directly.

#### 3. FP32 *regresses* at large N — the L2 wall

| N | FP32 cuBLAS | Δ vs N=4096 |
|---|---|---|
| 4096  | 27,509 | — |
| 8192  | 25,263 | **−8.2%** |
| 16384 | 23,691 | **−13.9%** |

At N=16384, each matrix is 1 GB — the working set blows past the 48 MB L2 cache by 60×. Tile reuse goes through DRAM (504 GB/s), and FP32 has the lowest arithmetic-intensity-per-byte of any precision path → it hits the bandwidth wall first. Tensor-core paths don't regress because each byte loaded feeds 4× more FLOPs (TF32) or 8× more (FP16/BF16).

#### 4. TF32 keeps scaling

TF32 climbs from 35.0 → 39.3 TFLOPS as N grows from 4096 → 16384 — opposite trend to FP32. Higher arithmetic intensity per byte means the L2 spillover hurts less. At N=16384, TF32 reaches **94% of FP32 peak / 54% of TF32 peak**.

#### 5. Small-N regime is launch-overhead-bound

At N=256 every cuBLAS kernel runs at 4–5 TFLOPS regardless of precision. The work is so small that kernel launch + tile staging dominates over actual FLOPs. TF32 is *faster* than FP16 at N=256 (4472 vs 4179) because cuBLAS picks a smaller-overhead microkernel for the FP32-layout path.

### Bonus: structured 2:4 sparsity ([extra/matmul_cuda_sparse.cu](../extra/matmul_cuda_sparse.cu))

Ada tensor cores can **skip the zeros** in a matrix that has exactly 2 zeros in every group of 4 consecutive elements per row — *structured 2:4 sparsity*. NVIDIA distributes a small library, **cuSPARSELt**, that compresses a dense FP16 matrix into the 2:4 format and runs sparse-aware tensor-core matmul on it. Theoretical peak: **2× dense FP16 ≈ 284 TFLOPS**.

Setup: separate library install (cuSPARSELt 0.7.1.0, ~340 MB tarball from NVIDIA's public redist archive), ~120 lines of API setup, and one important caveat — to measure this we have to **prune A to fit 2:4** (zeroing half its elements) before benchmarking. So the result is a hardware-ceiling measurement, not "speedup on the same problem". For real ML inference, weights are often pruned to 2:4 by design, so this *is* a real workload there.

| N | Sparse FP16 GFLOP/s | × dense FP16 (same N) | × CPU best |
|---|---|---|---|
| 1024  | 65,440  | 1.58× | 168× |
| 4096  | **124,611** | **1.70×** | **321×** |
| 8192  | 116,318 | 1.52× | 299× |
| 16384 | 82,500  | 1.07× | 212× |

GFLOP/s reported as effective dense rate (`2·N³ / time`), so the numbers are directly comparable to the dense FP16 row.

**Key findings:**

- **Peak at N=4096, not N=16384** — opposite of dense FP16. Sparse tensor cores chew tiles ~2× faster than dense, so the L2→register feed becomes the bottleneck *even sooner*. By N=16384, sparse is barely faster than dense.
- **124.6 TFLOPS = 44% of the 284 TFLOPS theoretical peak.** Same "library leaves headroom" pattern as every other tier we've measured. The remaining 56% is register-tile / async-copy / sparse-microkernel optimisations that even cuSPARSELt doesn't fully exploit on consumer Ada.
- **1.70× over dense FP16** at the sweet spot, not the theoretical 2×. Realistic ceiling for "ML inference with structured pruning" on this card.

### Performance ladder (best across all N)

| Implementation | GFLOP/s | × CPU best | % of relevant peak |
|---|---|---|---|
| CPU naive (1 thread) | 0.35 | 0.001× | 0.03% of 1075 |
| CPU best (16 threads, SMT) | 388.65 | 1× | 36% of 1075 |
| PyTorch MKL (16 threads) | 887.12 | 2.3× | 82% of 1075 |
| GPU naive | 2,132 | 5.5× | 5.8% of FP32 peak |
| GPU tiled (shared-memory) | 2,739 | 7.0× | 7.5% of FP32 peak |
| GPU cuBLAS FP32 (best, N=4096) | 27,509 | **71×** | 75% of FP32 peak |
| GPU cuBLAS TF32 (best, N=16384) | 39,324 | **101×** | 54% of TF32 peak |
| GPU cuBLAS FP16 (best, N=16384) | 77,381 | 199× | 54% of FP16 peak |
| GPU cuBLAS BF16 (best, N=16384) | 77,106 | 199× | 54% of FP16 peak |
| **GPU cuSPARSELt FP16 + 2:4 sparsity (best, N=4096)** | **124,611** | **321×** | **44% of sparse peak / 1.70× over dense FP16** |

### Why the remaining 46% to FP16 peak exists (parallel to Task 6)

| Technique | What it does | Why our kernel doesn't (and why even cuBLAS leaves it) |
|---|---|---|
| **Register tiling** | Each thread accumulates a 4×4 or 8×8 C-tile in registers across the full k-loop; C never spills to shared memory | Our hand-rolled kernel uses one `acc` register/thread. cuBLAS uses 4×4–8×8, but at FP16/BF16 the register file fills up before the tile-emit cadence catches up to peak |
| **Warp / thread-block tile hierarchy** | Outer block-tile + inner warp-tile + register-tile — 3-level GPU analogue of CPU's L3/L2/L1 tiling | We use a single 32×32 thread-block tile |
| **Software pipelining of `cp.async`** | Asynchronous global→shared loads overlap with compute on the previous tile (Ampere+ feature) | Our kernel uses synchronous loads + `__syncthreads()`. cuBLAS uses cp.async but its scheduling at FP16 isn't fully saturated |
| **Tensor-core path (TF32/FP16/BF16)** | 2×–4× CUDA-core peak via 4×4 matrix-matrix FMA per cycle on tensor cores | **Measured all four:** FP32 (kernels 3) → +0%; TF32 (kernel 4) → +28% at N=16384; FP16/BF16 (kernels 5–6) → +181% at N=16384 |
| **Structured 2:4 sparsity** | Pruning 2 of every 4 weights → tensor cores skip the zeros → another 2× on top of FP16 | **Measured (kernel 7, [extra/matmul_cuda_sparse.cu](../extra/matmul_cuda_sparse.cu))** — gave 124.6 TFLOPS at N=4096 = **1.70× over dense FP16 / 44% of theoretical sparse peak**. Synthetic for our random matrices (we prune A first); real for ML weight matrices |

Same structural lesson as CPU: the compiler-generic kernel hits ~7% of FP32 peak; the vendor-tuned library hits 75% (FP32) → 54% (TF32) → 54% (FP16). The gap on GPU is bigger than the 2.3× gap on CPU because GPUs leave *more* on the table for naive kernels — more parallelism, more cache layers, and entire alternative compute paths (tensor cores, sparsity) that scalar kernels can't access at all.

### Conclusion

The progressive-optimisation arc from Tasks 1–7 (memory access → SIMD → tiling → threading) compresses 1110× of speedup on CPU. Task 8 shows the GPU story has a similar shape but with multiple ceilings stacked on top:

```
CPU naive   0.35 GFLOP/s  →  CPU best          388.65 GFLOP/s    (+1110×)
                          →  GPU FP32 cuBLAS    27.5 TFLOPS       (+71× over CPU best)
                          →  GPU TF32 cuBLAS    39.3 TFLOPS       (+101×)
                          →  GPU FP16 cuBLAS    77.4 TFLOPS       (+199×)
                          →  GPU FP16 + 2:4 sp 124.6 TFLOPS       (+321×)   ← measured peak
```

Each tier is a fundamentally different kind of investment:

- **CPU 0 → 1110×:** algorithmic + compiler + cache + threading (Tasks 1–7). This is what a careful programmer can do without leaving C.
- **CPU 1110× → GPU 71×:** one architecture switch — call cuBLAS FP32 instead of writing a CPU kernel.
- **GPU 71× → 101×:** TF32 — change one math-mode flag. No precision loss for matmul accumulation.
- **GPU 101× → 199×:** FP16 / BF16 — accept ~10-bit-mantissa multiplies, use `cublasGemmEx`. This is what every ML training framework defaults to on Ampere+.
- **GPU 199× → 321×:** structured 2:4 sparsity — accept that A has half-zeros (real for pruned ML weights), use cuSPARSELt. The silicon literally skips the zeros.

The cumulative gap between "what a careful programmer writes from scratch" (our hand-rolled tiled kernel at 7% of FP32 peak) and "what the hardware can be coaxed to do" (sparsity at 44% of sparse peak) is **45,000×**. Most of that gap is **specialised hardware paths**, not algorithmic cleverness — tensor cores and sparsity are silicon, accessible only through vendor libraries.

The lesson generalises: **vendor microkernels are a structural moat, *specialised hardware paths* are a bigger one, and *precision/data-format tradeoffs* are bigger still**. For ML matmul on modern accelerators, *not* using the appropriate vendor path is leaving 100–500× of throughput on the table.

---

## Figures

> Place performance plots in `figures/` and reference them here.

![Performance comparison](figures/performance.png)
