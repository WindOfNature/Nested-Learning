"""GPU kernels using Triton. Requires torch + triton."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util


@dataclass
class TritonHandle:
    available: bool
    message: str


def available() -> TritonHandle:
    torch_spec = importlib.util.find_spec("torch")
    triton_spec = importlib.util.find_spec("triton")
    if torch_spec is None or triton_spec is None:
        return TritonHandle(False, "Triton not available: missing torch/triton")
    return TritonHandle(True, "Triton available")


def _load_triton():
    if not available().available:
        raise RuntimeError("Triton not available")
    import torch
    import triton
    import triton.language as tl
    return torch, triton, tl


def matmul(a, b):
    torch, triton, tl = _load_triton()

    @triton.jit
    def matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a_block = tl.load(
                a_ptr + (offs_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak),
                mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K),
                other=0.0,
            )
            b_block = tl.load(
                b_ptr + ((offs_k[:, None] + k) * stride_bk + offs_n[None, :] * stride_bn),
                mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(a_block, b_block)
        c = acc.to(tl.float16)
        tl.store(
            c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            c,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )

    a_t = a if isinstance(a, torch.Tensor) else torch.as_tensor(a, device="cuda")
    b_t = b if isinstance(b, torch.Tensor) else torch.as_tensor(b, device="cuda")
    if not a_t.is_cuda or not b_t.is_cuda:
        raise ValueError("Triton matmul requires CUDA tensors")
    M, K = a_t.shape
    K2, N = b_t.shape
    if K != K2:
        raise ValueError("Incompatible shapes")
    c_t = torch.empty((M, N), device="cuda", dtype=a_t.dtype)
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
    matmul_kernel[grid](
        a_t,
        b_t,
        c_t,
        M,
        N,
        K,
        a_t.stride(0),
        a_t.stride(1),
        b_t.stride(0),
        b_t.stride(1),
        c_t.stride(0),
        c_t.stride(1),
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=32,
    )
    return c_t


def layernorm(x, gamma, beta, eps: float = 1e-5):
    torch, triton, tl = _load_triton()

    @triton.jit
    def layernorm_kernel(x_ptr, gamma_ptr, beta_ptr, y_ptr, stride_xm, stride_xn, stride_ym, stride_yn, N, EPS):
        pid = tl.program_id(0)
        offs = tl.arange(0, N)
        x = tl.load(x_ptr + pid * stride_xm + offs * stride_xn, mask=offs < N, other=0.0)
        mean = tl.sum(x, axis=0) / N
        var = tl.sum((x - mean) * (x - mean), axis=0) / N
        inv = tl.rsqrt(var + EPS)
        gamma = tl.load(gamma_ptr + offs, mask=offs < N, other=1.0)
        beta = tl.load(beta_ptr + offs, mask=offs < N, other=0.0)
        y = (x - mean) * inv * gamma + beta
        tl.store(y_ptr + pid * stride_ym + offs * stride_yn, y, mask=offs < N)

    x_t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, device="cuda")
    if not x_t.is_cuda:
        raise ValueError("Triton layernorm requires CUDA tensors")
    gamma_t = gamma if isinstance(gamma, torch.Tensor) else torch.as_tensor(gamma, device="cuda")
    beta_t = beta if isinstance(beta, torch.Tensor) else torch.as_tensor(beta, device="cuda")
    y = torch.empty_like(x_t)
    grid = (x_t.shape[0],)
    layernorm_kernel[grid](
        x_t,
        gamma_t,
        beta_t,
        y,
        x_t.stride(0),
        x_t.stride(1),
        y.stride(0),
        y.stride(1),
        x_t.shape[1],
        eps,
    )
    return y
