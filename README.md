# perf-kernels-cpu

Matrix multiplication, row softmax and attention written three ways each: NumPy (BLAS), Numba, and a C extension. They are optimised step by step (loop order, tiling, threads, fused single-pass math), benchmarked against NumPy/BLAS and placed on a **measured roofline**.

## Setup
```bash
cd ml/perf-kernels-cpu
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt      # and a C compiler with OpenMP (gcc)
```

## Usage
```bash
python -m pytest -q        # 34 tests, ~10 s (builds the C library on first use)
python bench.py            # ~75 s → results/bench.md, results/bench.json, results/roofline.png
```

## Architecture
| file | content |
|---|---|
| `kernels/ref.py` | NumPy references: `A @ B` (BLAS), stable softmax, attention that materialises the n×m scores |
| `kernels/nb.py` | Numba: naive ijk, ikj, tiled + `prange`, fused parallel softmax, online-softmax attention |
| `kernels/kernels.c` | C: the same kernels, OpenMP for the parallel ones |
| `kernels/ckernels.py` | builds `kernels.c` with `cc -O3 -march=native -fopenmp` on first use (atomic, cached by mtime) and wraps it with `ctypes`, so there is no build system and no CPython C-API boilerplate |
| `bench.py` | one subprocess per thread count (1, 4), so BLAS/OpenMP/Numba pools are sized before import. Median of 3-5 runs after warm-up. Also measures the roofs |

Correctness tests run every implementation against NumPy:
- shapes that are not multiples of the 64 tile (1×1, 67×45×91, 130×129×65);
- softmax with logits around 1000, which overflow without max-subtraction;
- attention with peaked scores, which exercises the online-softmax rescaling.

## Results (4 vCPU Xeon @ 2.1 GHz, AVX-512, float32)
**Roofs (measured):** at 1 thread, 168 GFLOP/s (BLAS sgemm) and 22 GB/s (stream add, 3 × 128 MB arrays, 16 B/element counting the write-allocate); at 4 threads, 619 GFLOP/s and 65 GB/s.

![roofline](results/roofline.png)

Matmul (GFLOP/s, n = 512 / 2048):

| impl | 1 thread, 512 | 4 threads, 512 | 1 thread, 2048 | 4 threads, 2048 |
|---|---|---|---|---|
| numpy / BLAS | **151** | **427** | **168** | **619** |
| Numba naive ijk | 2.4 | 2.4 | - | - |
| C naive ijk | 2.5 | 2.5 | - | - |
| Numba ikj | 26.8 | 25.7 | 14.0 | 11.9 |
| C ikj | 22.3 | 24.6 | 9.3 | 11.4 |
| Numba tiled + prange | 16.7 | 74.7 | 16.3 | 65.5 |
| C tiled + OpenMP | 17.3 | 76.0 | 16.2 | 68.6 |

Row softmax 4096×1024 (ms): NumPy 10.7 (1 thread) / 12.2 (4 threads); Numba fused 13.9 / **3.6**; C fused 14.4 / **3.6**.
Attention n = m = 2048, d = 64 (ms): NumPy materialised 24.4 / **18.7**; C online 123 / 30.8; Numba online 170 / 40.9.
Full table: [results/bench.md](results/bench.md).

## Analysis
- **Loop order is the first 10×.** Naive ijk walks B down a column, so every element costs a new cache line. The reduction into one scalar also can't be vectorised without `-ffast-math`, because float addition order matters. ikj makes the inner loop `c[j] += a * b[j]` over contiguous rows, which both compilers vectorise with AVX-512: 2.5 → 22-27 GFLOP/s at n=512.
- **Cache size decides when tiling matters.** At n=512 (1 MB per matrix) ikj runs from cache. At n=2048 (16 MB per matrix) ikj drops to 9-14 GFLOP/s, streaming B from memory once per row of A. Tiling (64×64 blocks, 48 KB working set) keeps blocks in cache, so it is the fastest single-thread version at 2048 (16 GFLOP/s). Its bigger payoff is that it creates independent row bands to parallelise: 4 threads give 66-69 GFLOP/s, a 4.0-4.2× speed-up over 1-thread tiled, i.e. linear.
- **BLAS is still about 9× faster** (619 vs 69 GFLOP/s at 4 threads; 168 vs 16 at 1 thread, 10×). Our best kernel reaches ~11 % of the compute roof. BLAS additionally:
  - packs A and B panels into contiguous buffers;
  - keeps a register-blocked micro-kernel (e.g. 16×6 outputs in registers, FMA every cycle, so each loaded value is reused about 6-16×);
  - tunes block sizes to L1/L2/L3.
  Our inner loop does one load + one FMA + one store per element, which makes it load/store-bound, not FMA-bound. That gap is the lesson: arithmetic intensity at the *register* level, not just the cache level, is what reaches the roof.
- **Softmax.** On the roofline, softmax's compulsory traffic (8 B per element) would allow 0.62 FLOP/B × 65 GB/s ≈ 40 GFLOP/s. All versions are far below that because `exp` costs ~10-20 FLOPs, not the 1 counted, so softmax is compute-bound on `exp` here. Fusing into one parallel kernel wins 3.4× at 4 threads (3.6 ms vs 12.2 ms). NumPy does not scale with threads at all: its ufuncs are single-threaded, and it makes about 5 full passes with temporaries (max, subtract, exp, sum, divide).
- **Online attention vs BLAS.** The online-softmax kernels (FlashAttention's recurrence) never store the 2048×2048 score matrix: O(d) extra memory per query instead of O(n·m). But their dot products are scalar loops, while NumPy spends its time in two BLAS GEMMs. With 4 threads, C online is 1.6× slower than NumPy, and the 16 MB score matrix still fits in this machine's memory hierarchy. The online version wins when n·m no longer fits (long sequences), or when its inner products are blocked into GEMM-like tiles, which is what FlashAttention does on GPUs.
- **Serial kernels don't speed up with 4 threads** (naive, ikj: same or noisier). They have no parallel loop. In the 4-thread process, idle OpenBLAS/OpenMP worker threads may still spin on the shared vCPUs.

**Debugging note.** The first Numba tiled kernel ran at 2.9 GFLOP/s, 4× slower than its own untiled ikj. Two hypotheses were tested and rejected: 2-D indexing, and `parallel=True` outlining (serial was equally slow). The cause: the inner loop started at a runtime offset `j0`, so Numba could not prove `j ≥ 0` and kept its negative-index wraparound check, which blocks vectorisation. Slicing tile-local views and looping from 0 gave 17 GFLOP/s, a 6× speed-up.

Variance: ±10-30 % between runs and across container restarts on this shared VM (an earlier run measured 4-thread BLAS at 370-400 GFLOP/s, this one at 619). The roof uses the best BLAS measurement in the same run.

## Not done (time box)
Register-blocked micro-kernel / panel packing, explicit SIMD intrinsics, auto-tuned tile sizes, blocked (GEMM-tiled) online attention, perf-counter (cache miss) measurements.
