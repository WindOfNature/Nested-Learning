"""Kernel registry for CPU/GPU custom kernels (Triton-ready placeholders)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict


KernelFn = Callable[..., object]


@dataclass
class KernelRegistry:
    """Registry for custom kernels with CPU/GPU targets."""

    cpu_kernels: Dict[str, KernelFn] = field(default_factory=dict)
    gpu_kernels: Dict[str, KernelFn] = field(default_factory=dict)

    def register_cpu(self, name: str, fn: KernelFn) -> None:
        self.cpu_kernels[name] = fn

    def register_gpu(self, name: str, fn: KernelFn) -> None:
        self.gpu_kernels[name] = fn

    def get(self, name: str, device: str = "cpu") -> KernelFn:
        if device == "cpu":
            return self.cpu_kernels[name]
        if device == "gpu":
            return self.gpu_kernels[name]
        raise ValueError(f"Unknown device '{device}'.")


GLOBAL_KERNELS = KernelRegistry()
