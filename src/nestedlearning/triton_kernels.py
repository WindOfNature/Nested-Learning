"""Triton GPU kernels for Nested Learning (optional)."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

import numpy as np

from nestedlearning.kernels import GLOBAL_KERNELS


def _triton_available() -> bool:
    return importlib.util.find_spec("triton") is not None


def register_triton_kernels() -> None:
    """Register Triton GPU kernels if Triton is available."""
    if not _triton_available():
        return

    triton = importlib.import_module("triton")
    tl = importlib.import_module("triton.language")

    @triton.jit
    def _matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        m: tl.constexpr,
        n: tl.constexpr,
        k: tl.constexpr,
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
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_block in range(0, k, BLOCK_K):
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < m) & (offs_k[None, :] + k_block < k), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] + k_block < k) & (offs_n[None, :] < n), other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

    def triton_matmul(a: np.ndarray, b: np.ndarray) -> Any:
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("Triton matmul expects 2D arrays.")
        m, k = a.shape
        k2, n = b.shape
        if k != k2:
            raise ValueError("Triton matmul shape mismatch.")
        a_contig = np.ascontiguousarray(a.astype(np.float32))
        b_contig = np.ascontiguousarray(b.astype(np.float32))
        c = np.empty((m, n), dtype=np.float32)
        grid = (triton.cdiv(m, 128), triton.cdiv(n, 128))
        _matmul_kernel[grid](
            a_contig,
            b_contig,
            c,
            m,
            n,
            k,
            a_contig.strides[0] // a_contig.itemsize,
            a_contig.strides[1] // a_contig.itemsize,
            b_contig.strides[0] // b_contig.itemsize,
            b_contig.strides[1] // b_contig.itemsize,
            c.strides[0] // c.itemsize,
            c.strides[1] // c.itemsize,
            BLOCK_M=128,
            BLOCK_N=128,
            BLOCK_K=32,
        )
        return c

    GLOBAL_KERNELS.register_gpu("matmul", triton_matmul)
