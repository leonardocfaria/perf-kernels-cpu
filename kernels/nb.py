"""Numba kernels: naive, loop-reordered, tiled+parallel matmul; fused softmax; streaming attention."""
import math

import numba
import numpy as np
from numba import prange

TILE = 64  # 3 float32 tiles of 64x64 = 48 KB: fits L2 comfortably
# fast-math without nnan/ninf: attention starts its running max at -inf and relies on exp(-inf) = 0
FAST = {"contract", "afn", "reassoc", "nsz", "arcp"}


@numba.njit(cache=True)
def matmul_naive(A, B):
    M, K = A.shape
    N = B.shape[1]
    C = np.zeros((M, N), np.float32)
    for i in range(M):
        for j in range(N):
            s = np.float32(0)
            for k in range(K):
                s += A[i, k] * B[k, j]  # B walked down a column: one cache line per element
            C[i, j] = s
    return C


@numba.njit(cache=True)
def matmul_ikj(A, B):
    M, K = A.shape
    N = B.shape[1]
    C = np.zeros((M, N), np.float32)
    for i in range(M):
        for k in range(K):
            a = A[i, k]
            for j in range(N):  # unit stride over B and C rows: LLVM vectorises this loop
                C[i, j] += a * B[k, j]
    return C


@numba.njit(parallel=True, cache=True)
def matmul_tiled(A, B):
    M, K = A.shape
    N = B.shape[1]
    C = np.zeros((M, N), np.float32)
    for ti in prange((M + TILE - 1) // TILE):  # each thread owns a band of C rows: no write sharing
        i0 = ti * TILE
        for k0 in range(0, K, TILE):
            for j0 in range(0, N, TILE):
                je = min(j0 + TILE, N)
                for i in range(i0, min(i0 + TILE, M)):
                    c = C[i, j0:je]  # tile-local views, inner loop from 0: Numba can prove j >= 0, so it drops
                    for k in range(k0, min(k0 + TILE, K)):  # the negative-index wraparound check and vectorises
                        a = A[i, k]
                        b = B[k, j0:je]
                        for j in range(c.size):
                            c[j] += a * b[j]
    return C


@numba.njit(parallel=True, cache=True, fastmath=FAST)
def softmax(X):
    R, Cn = X.shape
    Y = np.empty_like(X)
    for r in prange(R):
        m = X[r, 0]
        for c in range(1, Cn):
            m = max(m, X[r, c])
        s = np.float32(0)
        for c in range(Cn):
            e = math.exp(X[r, c] - m)
            Y[r, c] = e
            s += e
        inv = np.float32(1) / s
        for c in range(Cn):
            Y[r, c] *= inv
    return Y


@numba.njit(parallel=True, cache=True, fastmath=FAST)
def attention(Q, K, V):
    """Online softmax per query row: running max m, running sum l, running output acc. Never stores n x m."""
    n, d = Q.shape
    m_keys = K.shape[0]
    O = np.empty((n, V.shape[1]), np.float32)
    scale = np.float32(1 / math.sqrt(d))
    for i in prange(n):
        m = np.float32(-np.inf)
        l = np.float32(0)
        acc = np.zeros(V.shape[1], np.float32)
        for j in range(m_keys):
            s = np.float32(0)
            for t in range(d):
                s += Q[i, t] * K[j, t]
            s *= scale
            if s > m:  # new max: rescale what was accumulated so far
                corr = math.exp(m - s)
                l *= corr
                acc *= corr
                m = s
            p = math.exp(s - m)
            l += p
            acc += p * V[j]
        O[i] = acc / l
    return O
