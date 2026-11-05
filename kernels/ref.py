"""NumPy references (matmul goes to the BLAS NumPy links against)."""
import numpy as np


def matmul(A, B):
    return A @ B


def softmax(X):
    e = np.exp(X - X.max(axis=1, keepdims=True))  # subtract the row max: exp never overflows
    return e / e.sum(axis=1, keepdims=True)


def attention(Q, K, V):
    """softmax(Q Kᵀ / sqrt(d)) V. Materialises the full n x m score matrix."""
    return softmax((Q @ K.T) / np.float32(np.sqrt(Q.shape[1]))) @ V
