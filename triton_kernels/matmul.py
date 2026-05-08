"""
Blocked matrix multiplication in OpenAI Triton.
Benchmark against torch.matmul (cuBLAS) and tune block sizes.
SDKs: Triton, PyTorch, W&B
"""
import torch
import triton
import triton.language as tl
from typing import Optional, Dict, Any


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Blocked GEMM: C = A @ B
    A: (M, K), B: (K, N), C: (M, N) — all float32
    Tuning knobs: BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M (super-grouping for L2 reuse)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Pointers to A and B tiles
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Accumulate in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store result
    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    BLOCK_M: int = 128,
    BLOCK_N: int = 256,
    BLOCK_K: int = 64,
    GROUP_M: int = 8,
) -> torch.Tensor:
    """Launch Triton matmul kernel."""
    assert A.shape[1] == B.shape[0], "Incompatible matrix dimensions"
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
    )
    return C


def autotune_matmul(M: int = 4096, N: int = 4096, K: int = 4096) -> Dict[str, Any]:
    """
    Sweep block size configs and find the fastest for given matrix shape.
    Returns best config + timing vs cuBLAS baseline.
    """
    import time
    device = "cuda"
    A = torch.randn(M, K, device=device, dtype=torch.float16)
    B = torch.randn(K, N, device=device, dtype=torch.float16)

    # cuBLAS baseline
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()
    cublas_ms = (time.perf_counter() - t0) / 20 * 1000
    cublas_tflops = 2 * M * N * K / cublas_ms / 1e9

    configs = [
        {"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_M": 8},
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
    ]

    best_config, best_ms, best_tflops = None, float("inf"), 0
    results = []

    for cfg in configs:
        try:
            # Warmup
            triton_matmul(A, B, **cfg)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                triton_matmul(A, B, **cfg)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 20 * 1000
            tflops = 2 * M * N * K / ms / 1e9
            speedup = cublas_ms / ms

            results.append({**cfg, "ms": round(ms, 3), "tflops": round(tflops, 2), "speedup_vs_cublas": round(speedup, 3)})
            print(f"  BLOCK={cfg['BLOCK_M']}x{cfg['BLOCK_N']}x{cfg['BLOCK_K']}: {ms:.2f}ms | {tflops:.1f} TFLOPS | {speedup:.2f}x vs cuBLAS")

            if ms < best_ms:
                best_ms, best_config, best_tflops = ms, cfg, tflops
        except Exception as e:
            print(f"  Config {cfg} failed: {e}")

    print(f"
cuBLAS baseline: {cublas_ms:.2f}ms | {cublas_tflops:.1f} TFLOPS")
    print(f"Best Triton config: {best_config} -> {best_ms:.2f}ms | {best_tflops:.1f} TFLOPS")
    return {
        "shape": (M, N, K),
        "cublas_ms": cublas_ms,
        "cublas_tflops": cublas_tflops,
        "best_config": best_config,
        "best_ms": best_ms,
        "best_tflops": best_tflops,
        "all_configs": results,
    }


@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused row-wise softmax — faster than two-pass PyTorch impl."""
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=col_offsets < n_cols)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4 if BLOCK_SIZE < 2048 else (8 if BLOCK_SIZE < 8192 else 16)
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](
        y, x,
        x.stride(0), y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
