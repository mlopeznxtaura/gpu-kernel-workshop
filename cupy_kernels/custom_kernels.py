"""
Raw CUDA C++ kernels via CuPy ElementwiseKernel and RawKernel.
Direct CUDA kernel authoring from Python — maximum control.
SDKs: CuPy, NumPy
"""
import numpy as np
from typing import Optional, Tuple

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np
    print("Warning: CuPy not available. Install: pip install cupy-cuda12x")


# ---- ElementwiseKernel: simple per-element ops ----

if CUPY_AVAILABLE:
    leaky_relu_kernel = cp.ElementwiseKernel(
        "float32 x, float32 alpha",
        "float32 y",
        "y = x > 0 ? x : alpha * x",
        "leaky_relu",
    )

    gelu_kernel = cp.ElementwiseKernel(
        "float32 x",
        "float32 y",
        """
        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;
        float cdf = 0.5f * (1.0f + tanhf(sqrt_2_over_pi * (x + coeff * x * x * x)));
        y = x * cdf;
        """,
        "gelu",
    )

    swish_kernel = cp.ElementwiseKernel(
        "float32 x",
        "float32 y",
        "y = x / (1.0f + expf(-x))",
        "swish",
    )

    layer_norm_kernel = cp.ElementwiseKernel(
        "float32 x, float32 mean, float32 inv_std, float32 gamma, float32 beta",
        "float32 y",
        "y = gamma * (x - mean) * inv_std + beta",
        "layer_norm_elementwise",
    )


# ---- RawKernel: full CUDA C++ control ----

FLASH_ATTENTION_KERNEL_SRC = r"""
extern "C" __global__
void flash_attention_forward(
    const float* __restrict__ Q,  // (B, H, S, D)
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ L,        // log-sum-exp per row
    const int B, const int H, const int S, const int D,
    const float scale,
    const int BLOCK_SIZE
) {
    // Each block handles one (b, h, i) row of Q
    int bid = blockIdx.x;
    int hid = blockIdx.y;
    int qid = blockIdx.z;   // query row index

    if (bid >= B || hid >= H || qid >= S) return;

    extern __shared__ float smem[];
    float* K_block = smem;
    float* V_block = smem + BLOCK_SIZE * D;

    // Accumulator for output and log-sum-exp
    float acc[64] = {0.0f};   // max D=64 in this simple impl
    float m_i = -1e9f;        // running max
    float l_i = 0.0f;         // running sum of exp

    int q_base = bid * H * S * D + hid * S * D + qid * D;

    for (int kv_start = 0; kv_start < S; kv_start += BLOCK_SIZE) {
        int kv_end = min(kv_start + BLOCK_SIZE, S);

        // Load K and V tiles into shared memory
        for (int i = threadIdx.x; i < (kv_end - kv_start) * D; i += blockDim.x) {
            int kv_row = kv_start + i / D;
            int d = i % D;
            int kv_base = bid * H * S * D + hid * S * D + kv_row * D;
            K_block[i] = (kv_row < S) ? K[kv_base + d] : 0.0f;
            V_block[i] = (kv_row < S) ? V[kv_base + d] : 0.0f;
        }
        __syncthreads();

        // Compute Q[qid] @ K_block.T -> attention scores
        for (int j = 0; j < kv_end - kv_start; j++) {
            float qk = 0.0f;
            for (int d = 0; d < D; d++) {
                qk += Q[q_base + d] * K_block[j * D + d];
            }
            qk *= scale;

            // Online softmax update (FlashAttention trick)
            float m_new = max(m_i, qk);
            float exp_qk = expf(qk - m_new);
            float exp_scale = expf(m_i - m_new);
            l_i = exp_scale * l_i + exp_qk;
            for (int d = 0; d < D; d++) {
                acc[d] = exp_scale * acc[d] + exp_qk * V_block[j * D + d];
            }
            m_i = m_new;
        }
        __syncthreads();
    }

    // Normalize and store output
    float inv_l = 1.0f / l_i;
    int o_base = bid * H * S * D + hid * S * D + qid * D;
    for (int d = 0; d < D; d++) {
        O[o_base + d] = acc[d] * inv_l;
    }
    // Store log-sum-exp for backward pass
    int l_base = bid * H * S + hid * S + qid;
    L[l_base] = m_i + logf(l_i);
}
"""

HISTOGRAM_KERNEL_SRC = r"""
extern "C" __global__
void histogram_atomic(
    const float* __restrict__ data,
    int* __restrict__ hist,
    const int n,
    const float min_val,
    const float max_val,
    const int n_bins
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    float val = data[tid];
    if (val < min_val || val >= max_val) return;
    int bin = (int)((val - min_val) / (max_val - min_val) * n_bins);
    bin = min(bin, n_bins - 1);
    atomicAdd(&hist[bin], 1);
}
"""

REDUCE_SUM_KERNEL_SRC = r"""
extern "C" __global__
void reduce_sum(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int n
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    sdata[tid] = (i < n ? input[i] : 0.0f) + (i + blockDim.x < n ? input[i + blockDim.x] : 0.0f);
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
"""


class CUDAKernelLibrary:
    """
    Collection of CuPy RawKernel wrappers for custom CUDA operations.
    """

    def __init__(self):
        if not CUPY_AVAILABLE:
            print("[CuPy] GPU not available — kernel library in stub mode")
            return
        self._histogram_kernel = cp.RawKernel(HISTOGRAM_KERNEL_SRC, "histogram_atomic")
        self._reduce_kernel = cp.RawKernel(REDUCE_SUM_KERNEL_SRC, "reduce_sum")
        print("[CuPy] Kernel library initialized")

    def leaky_relu(self, x: "cp.ndarray", alpha: float = 0.01) -> "cp.ndarray":
        if not CUPY_AVAILABLE:
            return np.where(x > 0, x, alpha * x)
        return leaky_relu_kernel(x, np.float32(alpha))

    def gelu(self, x: "cp.ndarray") -> "cp.ndarray":
        if not CUPY_AVAILABLE:
            return x * 0.5 * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))
        return gelu_kernel(x)

    def histogram_gpu(
        self, data: "cp.ndarray", n_bins: int = 256,
        min_val: float = 0.0, max_val: float = 1.0,
    ) -> "cp.ndarray":
        """GPU-accelerated histogram using atomic adds."""
        if not CUPY_AVAILABLE:
            return np.histogram(data, bins=n_bins, range=(min_val, max_val))[0]
        n = data.size
        hist = cp.zeros(n_bins, dtype=cp.int32)
        block = 256
        grid = (n + block - 1) // block
        self._histogram_kernel(
            (grid,), (block,),
            (data.astype(cp.float32), hist, np.int32(n),
             np.float32(min_val), np.float32(max_val), np.int32(n_bins))
        )
        return hist

    def reduce_sum_gpu(self, data: "cp.ndarray") -> float:
        """Parallel reduction sum using shared memory."""
        if not CUPY_AVAILABLE:
            return float(np.sum(data))
        block = 256
        n = data.size
        x = data.astype(cp.float32).ravel()
        while n > 1:
            grid = (n + block * 2 - 1) // (block * 2)
            out = cp.zeros(grid, dtype=cp.float32)
            self._reduce_kernel(
                (grid,), (block,), (x, out, np.int32(n)),
                shared_mem=block * 4
            )
            x = out
            n = grid
        return float(x[0])
