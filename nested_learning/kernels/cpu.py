"""CPU kernels using NumPy and optional Numba for optimized operations."""

from __future__ import annotations

import importlib.util
import numpy as np


def _numba_available() -> bool:
    return importlib.util.find_spec("numba") is not None


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


if _numba_available():
    from numba import njit, prange
else:  # pragma: no cover - fallback
    njit = None
    prange = range


if _torch_available():
    import torch
else:  # pragma: no cover - optional
    torch = None


def _matmul_fallback(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b


if njit:
    @njit(parallel=True, fastmath=True)
    def _matmul_numba(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        m, k = a.shape
        k2, n = b.shape
        if k != k2:
            raise ValueError("Incompatible shapes")
        out = np.zeros((m, n), dtype=a.dtype)
        for i in prange(m):
            for j in range(n):
                acc = 0.0
                for kk in range(k):
                    acc += a[i, kk] * b[kk, j]
                out[i, j] = acc
        return out
else:
    _matmul_numba = None


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if _matmul_numba is not None:
        return _matmul_numba(a, b)
    return _matmul_fallback(a, b)


def matmul_torch(a, b):
    if torch is None:
        raise RuntimeError("torch is required for matmul_torch")
    a_np = a.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    out = matmul(a_np, b_np)
    return torch.from_numpy(out).to(a.device)


if njit:
    @njit(parallel=True, fastmath=True)
    def _layernorm_numba(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float) -> np.ndarray:
        out = np.empty_like(x)
        for i in prange(x.shape[0]):
            mean = np.mean(x[i])
            var = np.var(x[i])
            inv = 1.0 / np.sqrt(var + eps)
            for j in range(x.shape[1]):
                out[i, j] = (x[i, j] - mean) * inv * gamma[j] + beta[j]
        return out
else:
    _layernorm_numba = None


def layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    if _layernorm_numba is not None:
        return _layernorm_numba(x, gamma, beta, eps)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def layernorm_torch(x, gamma, beta, eps: float = 1e-5):
    if torch is None:
        raise RuntimeError("torch is required for layernorm_torch")
    x_np = x.detach().cpu().numpy()
    gamma_np = gamma.detach().cpu().numpy()
    beta_np = beta.detach().cpu().numpy()
    out = layernorm(x_np, gamma_np, beta_np, eps)
    return torch.from_numpy(out).to(x.device)
