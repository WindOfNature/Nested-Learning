"""Minimal tensor and autograd utilities (no torch dependency)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from nestedlearning.kernels import GLOBAL_KERNELS

BackpropFn = Callable[[], None]


@dataclass
class SimpleTensor:
    """A minimal autograd-enabled tensor."""

    data: np.ndarray
    requires_grad: bool = True
    grad: np.ndarray | None = None
    device: str = "cpu"
    _backward: BackpropFn = field(default=lambda: None, repr=False)
    _prev: set["SimpleTensor"] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data, dtype=np.float32)
        if self.grad is None and self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def zero_grad(self) -> None:
        if self.grad is not None:
            self.grad = np.zeros_like(self.data)

    def backward(self) -> None:
        topo: list[SimpleTensor] = []
        visited: set[SimpleTensor] = set()

        def build(node: SimpleTensor) -> None:
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build(child)
                topo.append(node)

        build(self)
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def __add__(self, other: "SimpleTensor") -> "SimpleTensor":
        other = ensure_tensor(other)
        out = SimpleTensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            device=self.device,
        )

        def _backward() -> None:
            if self.grad is not None:
                self.grad = self.grad + out.grad
            if other.grad is not None:
                other.grad = other.grad + out.grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __matmul__(self, other: "SimpleTensor") -> "SimpleTensor":
        other = ensure_tensor(other)
        matmul_kernel = GLOBAL_KERNELS.get("matmul", device=self.device)
        out = SimpleTensor(
            matmul_kernel(self.data, other.data),
            requires_grad=self.requires_grad or other.requires_grad,
            device=self.device,
        )

        def _backward() -> None:
            if self.grad is not None:
                self.grad = self.grad + out.grad @ other.data.T
            if other.grad is not None:
                other.grad = other.grad + self.data.T @ out.grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __mul__(self, other: "SimpleTensor") -> "SimpleTensor":
        other = ensure_tensor(other)
        out = SimpleTensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            device=self.device,
        )

        def _backward() -> None:
            if self.grad is not None:
                self.grad = self.grad + other.data * out.grad
            if other.grad is not None:
                other.grad = other.grad + self.data * out.grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __sub__(self, other: "SimpleTensor") -> "SimpleTensor":
        return self + (ensure_tensor(other) * SimpleTensor(np.array(-1.0), requires_grad=False))

    def __neg__(self) -> "SimpleTensor":
        return self * SimpleTensor(np.array(-1.0), requires_grad=False)

    def relu(self) -> "SimpleTensor":
        out_data = np.maximum(0, self.data)
        out = SimpleTensor(out_data, requires_grad=self.requires_grad, device=self.device)

        def _backward() -> None:
            if self.grad is not None:
                self.grad = self.grad + (self.data > 0) * out.grad

        out._backward = _backward
        out._prev = {self}
        return out

    def sum(self) -> "SimpleTensor":
        out = SimpleTensor(self.data.sum(), requires_grad=self.requires_grad, device=self.device)

        def _backward() -> None:
            if self.grad is not None:
                self.grad = self.grad + np.ones_like(self.data) * out.grad

        out._backward = _backward
        out._prev = {self}
        return out


def ensure_tensor(value: SimpleTensor | np.ndarray | float) -> SimpleTensor:
    if isinstance(value, SimpleTensor):
        return value
    return SimpleTensor(np.array(value, dtype=np.float32), requires_grad=False)


def mse_loss(pred: SimpleTensor, target: SimpleTensor) -> SimpleTensor:
    diff = pred - target
    return (diff * diff).sum()
