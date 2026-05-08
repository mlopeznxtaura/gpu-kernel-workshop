# GPU Kernel Workshop

Cluster 14 of the NextAura 500 SDKs / 25 Clusters project.

Write, profile, and optimize custom GPU kernels from Python-level abstractions down to CUDA C++. From Triton's Python-like syntax to raw CuPy CUDA kernels.

## Architecture

- Triton (OpenAI) for high-performance Python-like GPU kernel authoring
- CuPy for launching raw CUDA C++ kernels from Python
- Numba for JIT-compiled GPU kernels with CUDA backend
- NVIDIA Warp for autodiff-capable GPU kernels
- Taichi for cross-backend GPU programming (CUDA, Metal, Vulkan)
- JAX for XLA-compiled GPU computation with vmap/jit
- cuBLAS/cuFFT/cuSPARSE via CuPy for production math primitives
- Nsight Compute profiling output parsed into W&B runs
- Plotly Dash for custom kernel profiling dashboard

## SDKs Used

CUDA Toolkit, Triton (OpenAI), CuPy SDK, Numba SDK, NVIDIA Warp, Taichi SDK, JAX, cuBLAS, cuFFT, cuSPARSE, SPIRV-Cross SDK, Slang SDK, OpenCL SDK, ROCm, Nsight Compute, RenderDoc SDK, Weights & Biases, Prometheus Client, FastAPI, Plotly Dash SDK

## Quickstart

```bash
pip install -r requirements.txt

# Benchmark Triton matmul vs cuBLAS
python main.py --mode triton --op matmul --M 4096 --N 4096 --K 4096

# Profile all kernels and log to W&B
python main.py --mode benchmark --log-wandb

# Launch profiling dashboard
python main.py --mode dash

# Run a specific kernel
python main.py --mode kernel --name flash_attention --batch 8 --seq 2048
```

## Structure

```
triton_kernels/   OpenAI Triton kernel implementations
cupy_kernels/     Raw CUDA C++ kernels via CuPy ElementwiseKernel / RawKernel
numba_kernels/    Numba CUDA JIT kernels
warp_kernels/     NVIDIA Warp autodiff kernels
benchmarks/       Kernel benchmarking harness vs cuBLAS baselines
profiling/        Nsight Compute output parser + W&B logger
dashboard/        Plotly Dash profiling timeline dashboard
main.py           CLI entry point
```
