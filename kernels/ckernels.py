"""Build kernels.c into a shared library on first use (cached by source mtime) and wrap it with ctypes."""
import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np

SRC = Path(__file__).with_name("kernels.c")
LIB = Path(__file__).with_name("_build") / "libkernels.so"
CFLAGS = ["-O3", "-march=native", "-fopenmp", "-shared", "-fPIC"]


def _load():
    if not LIB.exists() or LIB.stat().st_mtime < SRC.stat().st_mtime:
        LIB.parent.mkdir(exist_ok=True)
        tmp = LIB.with_suffix(f".{os.getpid()}.so")
        subprocess.run([os.environ.get("CC", "cc"), *CFLAGS, str(SRC), "-o", str(tmp), "-lm"], check=True)
        os.replace(tmp, LIB)  # atomic: parallel test workers never load a half-written library
    lib = ctypes.CDLL(str(LIB))
    f32 = np.ctypeslib.ndpointer(np.float32, ndim=2, flags="C_CONTIGUOUS")
    i = ctypes.c_int
    for name in ("matmul_naive", "matmul_ikj", "matmul_tiled"):
        getattr(lib, name).argtypes = [f32, f32, f32, i, i, i]
    lib.softmax.argtypes = [f32, f32, i, i]
    lib.attention.argtypes = [f32, f32, f32, f32, i, i, i, i]
    lib.set_threads.argtypes = [i]
    return lib


_lib = _load()


def _f32(*arrays):
    out = [np.ascontiguousarray(a, dtype=np.float32) for a in arrays]
    if any(a.ndim != 2 or 0 in a.shape for a in out):
        raise ValueError("expected non-empty 2-D arrays")
    return out


def set_threads(n: int):
    _lib.set_threads(n)


def _matmul(fn):
    def run(A, B):
        A, B = _f32(A, B)
        (M, K), N = A.shape, B.shape[1]
        if B.shape[0] != K:  # a real check, not assert: C would read past B's end
            raise ValueError(f"inner dimensions differ: {A.shape} @ {B.shape}")
        C = np.empty((M, N), np.float32)
        fn(A, B, C, M, N, K)
        return C
    return run


matmul_naive, matmul_ikj, matmul_tiled = (_matmul(getattr(_lib, n)) for n in ("matmul_naive", "matmul_ikj", "matmul_tiled"))


def softmax(X):
    (X,) = _f32(X)
    Y = np.empty_like(X)
    _lib.softmax(X, Y, *X.shape)
    return Y


def attention(Q, K, V):
    Q, K, V = _f32(Q, K, V)
    if Q.shape[1] != K.shape[1] or K.shape[0] != V.shape[0]:
        raise ValueError(f"shape mismatch: Q {Q.shape}, K {K.shape}, V {V.shape}")
    O = np.empty((Q.shape[0], V.shape[1]), np.float32)
    _lib.attention(Q, K, V, O, Q.shape[0], K.shape[0], Q.shape[1], V.shape[1])
    return O
