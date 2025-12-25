"""Kernel backends."""

from .cpu import matmul as cpu_matmul, layernorm as cpu_layernorm
from .gpu import matmul as gpu_matmul, layernorm as gpu_layernorm

__all__ = ["cpu_matmul", "cpu_layernorm", "gpu_matmul", "gpu_layernorm"]
