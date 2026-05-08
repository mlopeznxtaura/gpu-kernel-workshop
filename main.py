"""
gpu-kernel-workshop — Entry Point

Write, benchmark, and profile GPU kernels: Triton, CuPy, Numba, Warp.

Usage:
  python main.py --mode triton --op matmul --M 4096 --N 4096 --K 4096
  python main.py --mode benchmark --log-wandb
  python main.py --mode cupy --op gelu --N 10000000
  python main.py --mode numba --op matmul --M 512 --N 512
  python main.py --mode dash
"""
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="GPU Kernel Workshop")
    parser.add_argument("--mode", required=True,
                        choices=["triton", "benchmark", "cupy", "numba", "dash"])
    parser.add_argument("--op", default="matmul",
                        choices=["matmul", "softmax", "gelu", "relu", "histogram"])
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--log-wandb", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="./results")
    return parser.parse_args()


def mode_triton(args):
    import torch
    if not torch.cuda.is_available():
        print("CUDA not available. Triton requires a GPU.")
        return

    if args.op == "matmul":
        from triton_kernels.matmul import autotune_matmul
        print(f"
Benchmarking Triton matmul ({args.M}x{args.N}x{args.K})...")
        results = autotune_matmul(args.M, args.N, args.K)
        print(f"
Best config: {results['best_config']}")
        print(f"Best TFLOPS: {results['best_tflops']:.1f}")
        print(f"cuBLAS baseline: {results['cublas_tflops']:.1f} TFLOPS")

    elif args.op == "softmax":
        from triton_kernels.matmul import triton_softmax
        x = torch.randn(args.M, args.N, device=args.device, dtype=torch.float32)
        y_triton = triton_softmax(x)
        y_torch = torch.softmax(x, dim=1)
        max_err = (y_triton - y_torch).abs().max().item()
        print(f"Softmax max error vs PyTorch: {max_err:.2e}")


def mode_benchmark(args):
    from benchmarks.kernel_benchmark import KernelBenchmarkSuite
    suite = KernelBenchmarkSuite(
        device=args.device,
        log_wandb=args.log_wandb,
        output_dir=args.output,
    )
    print("
Benchmarking matmul kernels...")
    suite.bench_matmul(sizes=[512, 1024, 2048])
    print("
Benchmarking activation kernels...")
    suite.bench_activations()
    path = suite.save_results()
    suite.finish()
    print(f"
Done. Results: {path}")


def mode_cupy(args):
    from cupy_kernels.custom_kernels import CUDAKernelLibrary, CUPY_AVAILABLE
    if not CUPY_AVAILABLE:
        print("CuPy not available.")
        return
    import cupy as cp, numpy as np
    lib = CUDAKernelLibrary()

    if args.op == "gelu":
        x = cp.random.randn(args.N).astype(cp.float32)
        y = lib.gelu(x)
        print(f"GELU output: mean={float(y.mean()):.4f}, std={float(y.std()):.4f}")
    elif args.op == "histogram":
        x = cp.random.randn(args.N).astype(cp.float32)
        hist = lib.histogram_gpu(x, n_bins=64, min_val=-4.0, max_val=4.0)
        print(f"Histogram bins (first 10): {hist[:10].tolist()}")


def mode_numba(args):
    from numba_kernels.cuda_kernels import NumbaKernelRunner, NUMBA_AVAILABLE
    if not NUMBA_AVAILABLE:
        print("Numba not available.")
        return
    import numpy as np
    runner = NumbaKernelRunner()

    if args.op == "matmul":
        A = np.random.randn(args.M, args.K).astype(np.float32)
        B = np.random.randn(args.K, args.N).astype(np.float32)
        C = runner.matmul(A, B)
        ref = A @ B
        max_err = np.abs(C - ref).max()
        print(f"Numba matmul max error: {max_err:.2e}")
    elif args.op == "relu":
        x = np.random.randn(args.N).astype(np.float32)
        y = runner.relu(x)
        print(f"ReLU: {(y < 0).sum()} negative values (should be 0)")


def mode_dash(args):
    import subprocess
    subprocess.run(["python", "-m", "dash", "dashboard/app.py"])


def main():
    args = parse_args()
    print("=" * 60)
    print("  GPU Kernel Workshop")
    print(f"  Mode: {args.mode.upper()} | Op: {args.op} | Device: {args.device}")
    print("=" * 60)

    dispatch = {
        "triton": mode_triton,
        "benchmark": mode_benchmark,
        "cupy": mode_cupy,
        "numba": mode_numba,
        "dash": mode_dash,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
