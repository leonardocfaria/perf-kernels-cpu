| kernel | impl | size | threads | ms | GFLOP/s | AI (FLOP/B) |
|---|---|---|---|---|---|---|
| roof | stream add (bandwidth) | 16777216 | 1 | 11.48 | 1.5 | 0.08 |
| roof | stream add (bandwidth) | 16777216 | 4 | 3.00 | 5.6 | 0.08 |
| roof | BLAS sgemm (peak) | 2048 | 1 | 132.89 | 129.3 | 341.33 |
| roof | BLAS sgemm (peak) | 2048 | 4 | 43.18 | 397.9 | 341.33 |
| attention | C online | n=2048,d=64 | 1 | 131.15 | 8.2 | 512.00 |
| attention | C online | n=2048,d=64 | 4 | 34.64 | 31.0 | 512.00 |
| attention | numba online | n=2048,d=64 | 1 | 211.34 | 5.1 | 512.00 |
| attention | numba online | n=2048,d=64 | 4 | 49.42 | 21.7 | 512.00 |
| attention | numpy (materialised) | n=2048,d=64 | 1 | 28.77 | 37.3 | 512.00 |
| attention | numpy (materialised) | n=2048,d=64 | 4 | 19.05 | 56.4 | 512.00 |
| matmul | C ikj | 2048 | 1 | 1337.78 | 12.8 | 341.33 |
| matmul | C ikj | 2048 | 4 | 1368.60 | 12.6 | 341.33 |
| matmul | C tiled+omp | 2048 | 1 | 1295.75 | 13.3 | 341.33 |
| matmul | C tiled+omp | 2048 | 4 | 319.40 | 53.8 | 341.33 |
| matmul | numba ikj | 2048 | 1 | 1438.83 | 11.9 | 341.33 |
| matmul | numba ikj | 2048 | 4 | 1500.51 | 11.4 | 341.33 |
| matmul | numba tiled+par | 2048 | 1 | 1421.61 | 12.1 | 341.33 |
| matmul | numba tiled+par | 2048 | 4 | 367.54 | 46.7 | 341.33 |
| matmul | numpy/BLAS | 2048 | 1 | 90.49 | 189.9 | 341.33 |
| matmul | numpy/BLAS | 2048 | 4 | 46.36 | 370.5 | 341.33 |
| matmul | C ikj | 512 | 1 | 11.12 | 24.1 | 85.33 |
| matmul | C ikj | 512 | 4 | 15.02 | 17.9 | 85.33 |
| matmul | C naive | 512 | 1 | 97.20 | 2.8 | 85.33 |
| matmul | C naive | 512 | 4 | 165.95 | 1.6 | 85.33 |
| matmul | C tiled+omp | 512 | 1 | 14.25 | 18.8 | 85.33 |
| matmul | C tiled+omp | 512 | 4 | 6.45 | 41.6 | 85.33 |
| matmul | numba ikj | 512 | 1 | 9.34 | 28.7 | 85.33 |
| matmul | numba ikj | 512 | 4 | 14.38 | 18.7 | 85.33 |
| matmul | numba naive | 512 | 1 | 130.65 | 2.1 | 85.33 |
| matmul | numba naive | 512 | 4 | 149.96 | 1.8 | 85.33 |
| matmul | numba tiled+par | 512 | 1 | 12.50 | 21.5 | 85.33 |
| matmul | numba tiled+par | 512 | 4 | 8.38 | 32.0 | 85.33 |
| matmul | numpy/BLAS | 512 | 1 | 2.31 | 116.0 | 85.33 |
| matmul | numpy/BLAS | 512 | 4 | 0.75 | 357.7 | 85.33 |
| softmax | C fused | 4096x1024 | 1 | 17.08 | 1.2 | 0.62 |
| softmax | C fused | 4096x1024 | 4 | 4.31 | 4.9 | 0.62 |
| softmax | numba fused | 4096x1024 | 1 | 16.77 | 1.3 | 0.62 |
| softmax | numba fused | 4096x1024 | 4 | 4.32 | 4.8 | 0.62 |
| softmax | numpy | 4096x1024 | 1 | 13.03 | 1.6 | 0.62 |
| softmax | numpy | 4096x1024 | 4 | 12.75 | 1.6 | 0.62 |
