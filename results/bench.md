| kernel | impl | size | threads | ms | GFLOP/s | AI (FLOP/B) |
|---|---|---|---|---|---|---|
| roof | BLAS sgemm (peak) | 2048 | 1 | 104.03 | 165.2 | 341.33 |
| roof | BLAS sgemm (peak) | 2048 | 4 | 28.06 | 612.2 | 341.33 |
| roof | stream add (bandwidth) | 33554432 | 1 | 24.02 | 1.4 | 0.06 |
| roof | stream add (bandwidth) | 33554432 | 4 | 8.28 | 4.1 | 0.06 |
| attention | C online | n=2048,d=64 | 1 | 122.92 | 8.7 | 512.00 |
| attention | C online | n=2048,d=64 | 4 | 30.80 | 34.9 | 512.00 |
| attention | numba online | n=2048,d=64 | 1 | 170.37 | 6.3 | 512.00 |
| attention | numba online | n=2048,d=64 | 4 | 40.88 | 26.3 | 512.00 |
| attention | numpy (materialised) | n=2048,d=64 | 1 | 24.38 | 44.0 | 512.00 |
| attention | numpy (materialised) | n=2048,d=64 | 4 | 18.71 | 57.4 | 512.00 |
| matmul | C ikj | 2048 | 1 | 1844.31 | 9.3 | 341.33 |
| matmul | C ikj | 2048 | 4 | 1504.29 | 11.4 | 341.33 |
| matmul | C tiled+omp | 2048 | 1 | 1059.90 | 16.2 | 341.33 |
| matmul | C tiled+omp | 2048 | 4 | 250.47 | 68.6 | 341.33 |
| matmul | numba ikj | 2048 | 1 | 1227.81 | 14.0 | 341.33 |
| matmul | numba ikj | 2048 | 4 | 1442.70 | 11.9 | 341.33 |
| matmul | numba tiled+par | 2048 | 1 | 1052.22 | 16.3 | 341.33 |
| matmul | numba tiled+par | 2048 | 4 | 262.41 | 65.5 | 341.33 |
| matmul | numpy/BLAS | 2048 | 1 | 102.24 | 168.0 | 341.33 |
| matmul | numpy/BLAS | 2048 | 4 | 27.74 | 619.3 | 341.33 |
| matmul | C ikj | 512 | 1 | 12.06 | 22.3 | 85.33 |
| matmul | C ikj | 512 | 4 | 10.93 | 24.6 | 85.33 |
| matmul | C naive | 512 | 1 | 107.12 | 2.5 | 85.33 |
| matmul | C naive | 512 | 4 | 108.84 | 2.5 | 85.33 |
| matmul | C tiled+omp | 512 | 1 | 15.51 | 17.3 | 85.33 |
| matmul | C tiled+omp | 512 | 4 | 3.53 | 76.0 | 85.33 |
| matmul | numba ikj | 512 | 1 | 10.02 | 26.8 | 85.33 |
| matmul | numba ikj | 512 | 4 | 10.43 | 25.7 | 85.33 |
| matmul | numba naive | 512 | 1 | 113.86 | 2.4 | 85.33 |
| matmul | numba naive | 512 | 4 | 112.26 | 2.4 | 85.33 |
| matmul | numba tiled+par | 512 | 1 | 16.03 | 16.7 | 85.33 |
| matmul | numba tiled+par | 512 | 4 | 3.59 | 74.7 | 85.33 |
| matmul | numpy/BLAS | 512 | 1 | 1.78 | 151.2 | 85.33 |
| matmul | numpy/BLAS | 512 | 4 | 0.63 | 427.4 | 85.33 |
| softmax | C fused | 4096x1024 | 1 | 14.41 | 1.5 | 0.62 |
| softmax | C fused | 4096x1024 | 4 | 3.64 | 5.8 | 0.62 |
| softmax | numba fused | 4096x1024 | 1 | 13.93 | 1.5 | 0.62 |
| softmax | numba fused | 4096x1024 | 4 | 3.60 | 5.8 | 0.62 |
| softmax | numpy | 4096x1024 | 1 | 10.71 | 2.0 | 0.62 |
| softmax | numpy | 4096x1024 | 4 | 12.21 | 1.7 | 0.62 |
