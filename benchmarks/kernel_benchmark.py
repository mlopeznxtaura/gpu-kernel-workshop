"""
Kernel benchmarking harness — compare custom kernels vs cuBLAS/PyTorch baselines.
Logs timing, TFLOPS, and memory bandwidth to W&B.
SDKs: PyTorch, Triton, CuPy, W&B
"""
import time
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import wandb


@dataclass
class BenchmarkResult:
    kernel_name: str
    op: str
    shape: Dict[str, int]
    mean_ms: float
    p50_ms: float
    p95_ms: float
    tflops: Optional[float]
    bandwidth_gbps: Optional[float]
    vs_baseline_speedup: Optional[float] = None
    config: Dict[str, Any] = field(default_factory=dict)
    device: str = "cuda"


def time_kernel(
    fn: Callable,
    n_warmup: int = 10,
    n_iters: int = 100,
    sync: bool = True,
) -> Dict[str, float]:
    """Accurately time a GPU kernel using CUDA events."""
    if not torch.cuda.is_available():
        t0 = time.perf_counter()
        for _ in range(n_iters):
            fn()
        elapsed = (time.perf_counter() - t0) / n_iters * 1000
        return {"mean_ms": elapsed, "p50_ms": elapsed, "p95_ms": elapsed}

    # Warmup
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    # Benchmark with CUDA events
    times = []
    for _ in range(n_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times = sorted(times)
    return {
        "mean_ms": float(np.mean(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
    }


class KernelBenchmarkSuite:
    """
    Run all kernel benchmarks and log results to W&B + local JSON.
    """

    def __init__(
        self,
        device: str = "cuda",
        log_wandb: bool = True,
        output_dir: str = "./benchmark_results",
        wandb_project: str = "gpu-kernel-workshop",
    ):
        self.device = device
        self.log_wandb = log_wandb
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []

        if log_wandb:
            self.run = wandb.init(project=wandb_project, name="kernel-benchmark")

    def bench_matmul(self, sizes: List[int] = [512, 1024, 2048, 4096]) -> List[BenchmarkResult]:
        """Benchmark Triton matmul vs cuBLAS across matrix sizes."""
        results = []
        for N in sizes:
            A = torch.randn(N, N, device=self.device, dtype=torch.float16)
            B = torch.randn(N, N, device=self.device, dtype=torch.float16)
            flops = 2 * N ** 3

            # cuBLAS baseline
            baseline_times = time_kernel(lambda: torch.matmul(A, B))
            baseline_tflops = flops / baseline_times["mean_ms"] / 1e9

            # Triton
            try:
                from triton_kernels.matmul import triton_matmul
                triton_times = time_kernel(lambda: triton_matmul(A, B))
                triton_tflops = flops / triton_times["mean_ms"] / 1e9
                speedup = baseline_times["mean_ms"] / triton_times["mean_ms"]

                r = BenchmarkResult(
                    kernel_name="triton_matmul",
                    op="matmul",
                    shape={"M": N, "N": N, "K": N},
                    mean_ms=triton_times["mean_ms"],
                    p50_ms=triton_times["p50_ms"],
                    p95_ms=triton_times["p95_ms"],
                    tflops=triton_tflops,
                    bandwidth_gbps=None,
                    vs_baseline_speedup=speedup,
                )
                results.append(r)
                self.results.append(r)
                print(f"  matmul N={N}: Triton={triton_times['mean_ms']:.2f}ms ({triton_tflops:.1f}T) | "
                      f"cuBLAS={baseline_times['mean_ms']:.2f}ms ({baseline_tflops:.1f}T) | "
                      f"speedup={speedup:.2f}x")

                if self.log_wandb:
                    wandb.log({
                        f"matmul/N{N}/triton_ms": triton_times["mean_ms"],
                        f"matmul/N{N}/cublas_ms": baseline_times["mean_ms"],
                        f"matmul/N{N}/triton_tflops": triton_tflops,
                        f"matmul/N{N}/speedup": speedup,
                    })
            except Exception as e:
                print(f"  Triton matmul N={N} failed: {e}")
        return results

    def bench_activations(self, N: int = 10_000_000) -> List[BenchmarkResult]:
        """Benchmark custom activation kernels vs PyTorch."""
        results = []
        x = torch.randn(N, device=self.device, dtype=torch.float32)
        bytes_rw = x.nbytes * 2  # read + write

        ops = {
            "relu": (lambda: torch.relu(x), None),
            "gelu": (lambda: torch.nn.functional.gelu(x), None),
        }

        try:
            from cupy_kernels.custom_kernels import CUDAKernelLibrary, CUPY_AVAILABLE
            if CUPY_AVAILABLE:
                import cupy as cp
                x_cp = cp.asarray(x.cpu().numpy())
                lib = CUDAKernelLibrary()
                ops["cupy_gelu"] = (lambda: lib.gelu(x_cp), None)
        except Exception:
            pass

        for name, (fn, _) in ops.items():
            times = time_kernel(fn)
            bw = bytes_rw / times["mean_ms"] / 1e6  # GB/s
            r = BenchmarkResult(
                kernel_name=name, op="activation",
                shape={"N": N},
                mean_ms=times["mean_ms"], p50_ms=times["p50_ms"], p95_ms=times["p95_ms"],
                tflops=None, bandwidth_gbps=bw,
            )
            results.append(r)
            self.results.append(r)
            print(f"  {name}: {times['mean_ms']:.3f}ms | {bw:.1f} GB/s")
        return results

    def save_results(self) -> str:
        path = str(self.output_dir / "benchmark_results.json")
        with open(path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        print(f"[Benchmark] Results saved: {path}")
        if self.log_wandb:
            wandb.save(path)
        return path

    def finish(self):
        if self.log_wandb:
            self.run.finish()
