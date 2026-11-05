// float32, row-major. Built by ckernels.py: cc -O3 -march=native -fopenmp -shared -fPIC
#include <math.h>
#include <omp.h>
#include <string.h>

#define TILE 64

void set_threads(int n) { omp_set_num_threads(n); }

// C[M,N] = A[M,K] @ B[K,N]. Naive: the k-loop strides through B by N floats (cache-hostile), and the
// reduction into s is not vectorised without -ffast-math (float addition order matters).
void matmul_naive(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += A[(size_t)i * K + k] * B[(size_t)k * N + j];
            C[(size_t)i * N + j] = s;
        }
}

// ikj: the inner loop is c[j] += a * b[j] over contiguous rows -> vectorised (AVX-512) by the compiler.
void matmul_ikj(const float *A, const float *B, float *C, int M, int N, int K) {
    memset(C, 0, sizeof(float) * (size_t)M * N);
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            const float a = A[(size_t)i * K + k];
            const float *b = B + (size_t)k * N;
            float *c = C + (size_t)i * N;
            for (int j = 0; j < N; j++) c[j] += a * b[j];
        }
}

// Tiled for cache reuse + OpenMP over row bands (each thread writes its own rows of C).
void matmul_tiled(const float *A, const float *B, float *C, int M, int N, int K) {
    memset(C, 0, sizeof(float) * (size_t)M * N);
#pragma omp parallel for schedule(static)
    for (int i0 = 0; i0 < M; i0 += TILE)
        for (int k0 = 0; k0 < K; k0 += TILE)
            for (int j0 = 0; j0 < N; j0 += TILE) {
                int ie = i0 + TILE < M ? i0 + TILE : M, ke = k0 + TILE < K ? k0 + TILE : K,
                    je = j0 + TILE < N ? j0 + TILE : N;
                for (int i = i0; i < ie; i++)
                    for (int k = k0; k < ke; k++) {
                        const float a = A[(size_t)i * K + k];
                        const float *b = B + (size_t)k * N;
                        float *c = C + (size_t)i * N;
                        for (int j = j0; j < je; j++) c[j] += a * b[j];
                    }
            }
}

// Numerically stable row softmax, fused: 3 passes over a row that stays in L1/L2.
void softmax(const float *X, float *Y, int R, int Cn) {
#pragma omp parallel for schedule(static)
    for (int r = 0; r < R; r++) {
        const float *x = X + (size_t)r * Cn;
        float *y = Y + (size_t)r * Cn;
        float m = x[0];
        for (int c = 1; c < Cn; c++) m = x[c] > m ? x[c] : m;
        float s = 0;
        for (int c = 0; c < Cn; c++) { y[c] = expf(x[c] - m); s += y[c]; }
        const float inv = 1.0f / s;
        for (int c = 0; c < Cn; c++) y[c] *= inv;
    }
}

// softmax(Q K^T / sqrt(d)) V with the online-softmax recurrence: O(d) memory per query, no n x m matrix.
void attention(const float *Q, const float *K, const float *V, float *O, int n, int m, int d, int dv) {
    const float scale = 1.0f / sqrtf((float)d);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float mx = -INFINITY, l = 0;
        float *acc = O + (size_t)i * dv;
        for (int t = 0; t < dv; t++) acc[t] = 0;
        for (int j = 0; j < m; j++) {
            float s = 0;
            for (int t = 0; t < d; t++) s += Q[(size_t)i * d + t] * K[(size_t)j * d + t];
            s *= scale;
            if (s > mx) {
                const float corr = expf(mx - s);
                l *= corr;
                for (int t = 0; t < dv; t++) acc[t] *= corr;
                mx = s;
            }
            const float p = expf(s - mx);
            l += p;
            for (int t = 0; t < dv; t++) acc[t] += p * V[(size_t)j * dv + t];
        }
        for (int t = 0; t < dv; t++) acc[t] /= l;
    }
}
