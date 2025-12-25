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
    if not x.is_cuda:
        raise ValueError("Triton layernorm requires CUDA tensors")
    if x.dim() < 2:
        raise ValueError("LayerNorm expects at least 2D input")
    n_features = x.shape[-1]
    if n_features > 4096:
        return torch.nn.functional.layer_norm(x, (n_features,), gamma, beta, eps)
    x_2d = x.reshape(-1, n_features)
    out = torch.empty_like(x_2d)
    BLOCK = triton.next_power_of_2(n_features)

    @triton.jit
    def layernorm_kernel(
        x_ptr,
        y_ptr,
        w_ptr,
        b_ptr,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        n_elements,
        eps,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_elements
        x = tl.load(x_ptr + row * stride_xm + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / n_elements
        var = tl.sum((x - mean) * (x - mean), axis=0) / n_elements
        rstd = tl.rsqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(y_ptr + row * stride_ym + cols * stride_yn, y, mask=mask)

    grid = (x_2d.shape[0],)
    layernorm_kernel[grid](
        x_2d,
        out,
        gamma,
        beta,
        x_2d.stride(0),
        x_2d.stride(1),
        out.stride(0),
        out.stride(1),
        n_features,
        eps,
        BLOCK=BLOCK,
    )
    return out.reshape_as(x)
