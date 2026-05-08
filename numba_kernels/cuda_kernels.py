"""
Numba CUDA JIT kernels — Python decorated GPU functions.
Matrix ops, reductions, and custom activations via Numba.
SDKs: Numba, NumPy
"""
import math
import numpy as np
from typing import Tuple

try:
    from numba import cuda, float32, int32
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: Numba not available. Install: pip install numba")


if NUMBA_AVAILABLE:

    @cuda.jit
    def vector_add_kernel(a, b, c):
        """GPU vector addition: c = a + b"""
        idx = cuda.grid(1)
        if idx < a.size:
            c[idx] = a[idx] + b[idx]

    @cuda.jit
    def relu_inplace_kernel(x):
        """In-place ReLU activation."""
        idx = cuda.grid(1)
        if idx < x.size:
            if x[idx] < 0.0:
                x[idx] = 0.0

    @cuda.jit
    def matmul_shared_kernel(A, B, C, TILE_SIZE):
        """
        Tiled matrix multiplication using shared memory.
        Reduces global memory bandwidth via TILE_SIZE x TILE_SIZE tiles.
        """
        sA = cuda.shared.array(shape=(32, 32), dtype=float32)
        sB = cuda.shared.array(shape=(32, 32), dtype=float32)

        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y

        row = by * TILE_SIZE + ty
        col = bx * TILE_SIZE + tx

        acc = float32(0.0)
        for k in range((A.shape[1] + TILE_SIZE - 1) // TILE_SIZE):
            if row < A.shape[0] and k * TILE_SIZE + tx < A.shape[1]:
                sA[ty, tx] = A[row, k * TILE_SIZE + tx]
            else:
                sA[ty, tx] = float32(0.0)

            if col < B.shape[1] and k * TILE_SIZE + ty < B.shape[0]:
                sB[ty, tx] = B[k * TILE_SIZE + ty, col]
            else:
                sB[ty, tx] = float32(0.0)

            cuda.syncthreads()
            for i in range(TILE_SIZE):
                acc += sA[ty, i] * sB[i, tx]
            cuda.syncthreads()

        if row < C.shape[0] and col < C.shape[1]:
            C[row, col] = acc

    @cuda.reduce
    def gpu_sum(a, b):
        return a + b

    @cuda.jit
    def conv2d_kernel(input, kernel, output, stride, padding):
        """
        2D convolution kernel — single input/output channel.
        input: (H, W), kernel: (KH, KW), output: (OH, OW)
        """
        oh = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        ow = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        OH, OW = output.shape
        KH, KW = kernel.shape
        H, W = input.shape

        if oh >= OH or ow >= OW:
            return

        acc = float32(0.0)
        for kh in range(KH):
            for kw in range(KW):
                ih = oh * stride - padding + kh
                iw = ow * stride - padding + kw
                if 0 <= ih < H and 0 <= iw < W:
                    acc += input[ih, iw] * kernel[kh, kw]
        output[oh, ow] = acc


class NumbaKernelRunner:
    """
    Launch Numba CUDA kernels with auto-configured grids.
    """

    def vector_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not NUMBA_AVAILABLE:
            return a + b
        n = a.size
        d_a = cuda.to_device(a.astype(np.float32))
        d_b = cuda.to_device(b.astype(np.float32))
        d_c = cuda.device_array(n, dtype=np.float32)
        threads = 256
        blocks = (n + threads - 1) // threads
        vector_add_kernel[blocks, threads](d_a, d_b, d_c)
        return d_c.copy_to_host()

    def relu(self, x: np.ndarray) -> np.ndarray:
        if not NUMBA_AVAILABLE:
            return np.maximum(x, 0)
        d_x = cuda.to_device(x.astype(np.float32))
        n = x.size
        threads = 256
        blocks = (n + threads - 1) // threads
        relu_inplace_kernel[blocks, threads](d_x)
        return d_x.copy_to_host()

    def matmul(self, A: np.ndarray, B: np.ndarray, tile_size: int = 32) -> np.ndarray:
        if not NUMBA_AVAILABLE:
            return A @ B
        M, K = A.shape
        K2, N = B.shape
        assert K == K2
        d_A = cuda.to_device(A.astype(np.float32))
        d_B = cuda.to_device(B.astype(np.float32))
        d_C = cuda.device_array((M, N), dtype=np.float32)
        threads = (tile_size, tile_size)
        blocks = ((N + tile_size - 1) // tile_size, (M + tile_size - 1) // tile_size)
        matmul_shared_kernel[blocks, threads](d_A, d_B, d_C, tile_size)
        return d_C.copy_to_host()

    def reduce_sum(self, x: np.ndarray) -> float:
        if not NUMBA_AVAILABLE:
            return float(np.sum(x))
        d_x = cuda.to_device(x.astype(np.float32))
        return float(gpu_sum(d_x))
