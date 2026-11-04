# perf-kernels-cpu

## Goal
Build "Performance engineering for ML ops". Implement and optimise matrix multiplication and softmax/attention in NumPy, Numba and a C extension, with tiling and vectorisation; benchmark against NumPy/BLAS and write a roofline-style analysis. Shows HPC skills relevant to ML systems. Keep it CPU-friendly (the cloud box has no GPU): small models and datasets, a fixed random seed, and a documented runtime.

## Definition of done
- [ ] correctness tests pass for all implementations
- [ ] benchmark table and analysis committed
- [ ] Tests pass with one command; CI workflow runs them
- [ ] README with setup, usage, architecture and results
- [ ] Self code review done
- [ ] Draft PR open

