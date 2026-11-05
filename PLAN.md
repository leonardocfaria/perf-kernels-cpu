# perf-kernels-cpu Implementation Plan

## Brainstorm (answered from SPEC + repo evidence)
- **What is it?** A hands-on study of why BLAS is fast: the same kernels (matmul, row softmax, attention) are written naively and then optimised (loop order for contiguous access, tiling for cache reuse, multi-threading, fused single-pass math) in Numba and in C. Each is benchmarked against NumPy/BLAS and placed on a measured roofline.
- **C extension:** a plain C file compiled on first use with `cc -O3 -march=native -fopenmp` into a shared library and called through `ctypes` (stdlib). No CPython C-API boilerplate, no build system.
- **Kernels** (float32, row-major):
  - matmul: naive ijk, ikj (unit-stride inner loop, so the compiler vectorises it), tiled + parallel. Numba and C. Reference: `A @ B` (BLAS).
  - softmax over rows: numerically stable, fused (max, exp-sum, scale), parallel rows. NumPy, Numba, C.
  - attention softmax(QKᵀ/√d)V: NumPy materialises the n×m score matrix. Numba and C use a streaming "online softmax" per query row (FlashAttention's idea), so they never store the matrix.
- **Roofline:** measured, not datasheet. Peak compute = large BLAS sgemm GFLOP/s at all threads; bandwidth = STREAM-style add on 64 MB arrays. Each kernel's arithmetic intensity (FLOP/byte of compulsory traffic) and achieved GFLOP/s are plotted against min(peak, AI·BW).
- **Non-goals:** GPU, hand-written intrinsics/assembly, auto-tuning tile sizes.

## Files
- `kernels/kernels.c`, `kernels/ckernels.py` (build + ctypes wrappers), `kernels/nb.py` (Numba), `kernels/ref.py` (NumPy).
- `bench.py` → `results/bench.md`, `results/roofline.png`, `results/bench.json`.

## Tasks (TDD)
- [x] 1. reference NumPy kernels + correctness tests on awkward shapes (not multiples of tiles), softmax stability with large logits.
- [x] 2. Numba kernels pass the same tests.
- [x] 3. C kernels (build on import) pass the same tests.
- [x] 4. bench + roofline + analysis; README; CI (gcc is on ubuntu-latest).
