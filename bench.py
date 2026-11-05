"""Benchmark every kernel at 1 and 4 threads, measure the machine's roofline, write results/.

python bench.py            (~2 min)     python bench.py --quick   (tiny sizes, smoke test)
Each thread count runs in a fresh subprocess so BLAS/OpenMP/Numba pools are sized before import.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def timeit(fn, *args, reps=5):
    fn(*args)  # warm-up (and Numba JIT)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn(*args)
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def run_one(threads: int, quick: bool) -> list[dict]:
    import numba
    import numpy as np

    from kernels import ckernels, nb, ref

    numba.set_num_threads(threads)
    ckernels.set_threads(threads)
    rng = np.random.default_rng(0)
    r32 = lambda *s: rng.standard_normal(s).astype(np.float32)
    rows = []

    def add(kernel, impl, size, seconds, flops, bytes_):
        rows.append({"threads": threads, "kernel": kernel, "impl": impl, "size": size, "ms": seconds * 1e3,
                     "gflops": flops / seconds / 1e9, "gbps": bytes_ / seconds / 1e9, "ai": flops / bytes_})

    # machine roofs: practical peak (large sgemm) and memory bandwidth (a = b + c on 64 MB arrays)
    n = 256 if quick else 2048
    A, B = r32(n, n), r32(n, n)
    add("roof", "BLAS sgemm (peak)", n, timeit(ref.matmul, A, B), 2 * n**3, 12 * n * n)
    m = 1 << (16 if quick else 24)
    b, c, a = r32(m), r32(m), np.empty(m, np.float32)
    add("roof", "stream add (bandwidth)", m, timeit(nb_stream_add(numba), a, b, c), m, 12 * m)

    for n, impls in ((128 if quick else 512, "all"), (256 if quick else 2048, "fast")):
        A, B = r32(n, n), r32(n, n)
        for name, fn in (("numpy/BLAS", ref.matmul), ("numba naive", nb.matmul_naive), ("numba ikj", nb.matmul_ikj),
                         ("numba tiled+par", nb.matmul_tiled), ("C naive", ckernels.matmul_naive),
                         ("C ikj", ckernels.matmul_ikj), ("C tiled+omp", ckernels.matmul_tiled)):
            if impls == "fast" and "naive" in name:
                continue  # minutes at 2048: the 512 row already shows the point
            add("matmul", name, n, timeit(fn, A, B, reps=3), 2 * n**3, 12 * n * n)

    R, C = (256, 256) if quick else (4096, 1024)
    X = r32(R, C)
    for name, fn in (("numpy", ref.softmax), ("numba fused", nb.softmax), ("C fused", ckernels.softmax)):
        add("softmax", name, f"{R}x{C}", timeit(fn, X), 5 * R * C, 8 * R * C)  # ~5 flops/elem (exp as 1)

    n, d = (128, 32) if quick else (2048, 64)
    Q, K, V = r32(n, d), r32(n, d), r32(n, d)
    for name, fn in (("numpy (materialised)", ref.attention), ("numba online", nb.attention), ("C online", ckernels.attention)):
        add("attention", name, f"n={n},d={d}", timeit(fn, Q, K, V), 4 * n * n * d, 4 * 4 * n * d)
    return rows


def nb_stream_add(numba):
    @numba.njit(parallel=True, cache=True)
    def stream_add(a, b, c):
        for i in numba.prange(a.size):
            a[i] = b[i] + c[i]
    return stream_add


def roofline_plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 6))
    ai = np.logspace(-1.5, 3, 200)
    colors = {1: "tab:blue", 4: "tab:red"}
    for t in (1, 4):
        peak = max(r["gflops"] for r in rows if r["threads"] == t and "BLAS" in r["impl"])  # best sgemm seen
        bw = next(r["gbps"] for r in rows if r["threads"] == t and r["impl"].startswith("stream"))
        ax.plot(ai, np.minimum(peak, ai * bw), color=colors[t], lw=2, label=f"roof {t} thread(s): {peak:.0f} GFLOP/s, {bw:.0f} GB/s")
        for r in rows:
            if r["threads"] == t and r["kernel"] != "roof":
                ax.scatter(r["ai"], r["gflops"], color=colors[t], s=25)
                if t == 4:
                    ax.annotate(f"{r['kernel']}: {r['impl']} ({r['size']})", (r["ai"], r["gflops"]), fontsize=6,
                                xytext=(4, 2), textcoords="offset points")
    ax.set(xscale="log", yscale="log", xlabel="arithmetic intensity (FLOP / byte of compulsory traffic)",
           ylabel="achieved GFLOP/s", title="Roofline (measured roofs; dots = kernels, red = 4 threads)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main(quick=False, out="results"):
    rows = []
    for t in (1, 4):
        env = {**os.environ, **{k: str(t) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}}
        res = subprocess.run([sys.executable, __file__, "--worker", str(t)] + (["--quick"] if quick else []),
                             env=env, capture_output=True, text=True, check=True, cwd=Path(__file__).parent)
        rows += json.loads(res.stdout.strip().splitlines()[-1])
    out = Path(out)
    out.mkdir(exist_ok=True)
    (out / "bench.json").write_text(json.dumps(rows, indent=2))
    lines = ["| kernel | impl | size | threads | ms | GFLOP/s | AI (FLOP/B) |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['kernel']} | {r['impl']} | {r['size']} | {r['threads']} | {r['ms']:.2f} | {r['gflops']:.1f} | {r['ai']:.2f} |"
              for r in sorted(rows, key=lambda r: (r["kernel"] != "roof", r["kernel"], str(r["size"]), r["impl"], r["threads"]))]
    (out / "bench.md").write_text("\n".join(lines) + "\n")
    roofline_plot(rows, out / "roofline.png")
    print("\n".join(lines))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--worker", type=int)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    if a.worker:
        print(json.dumps(run_one(a.worker, a.quick)))
    else:
        main(a.quick, a.out)
