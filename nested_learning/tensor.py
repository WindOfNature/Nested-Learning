"""Autograd tensor implementation inspired by nested learning principles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Set

import numpy as np


@dataclass
class _Context:
    parents: Sequence["Tensor"]
    backward: Callable[[np.ndarray], None]


class Tensor:
    _rng = np.random.default_rng()

    def __init__(self, data, requires_grad: bool = False, name: Optional[str] = None):
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._ctx: Optional[_Context] = None
        self.name = name

    @staticmethod
    def set_seed(seed: int):
        Tensor._rng = np.random.default_rng(seed)

    @staticmethod
    def zeros(shape, requires_grad: bool = False, name: Optional[str] = None):
        return Tensor(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad, name=name)

    @staticmethod
    def ones(shape, requires_grad: bool = False, name: Optional[str] = None):
        return Tensor(np.ones(shape, dtype=np.float32), requires_grad=requires_grad, name=name)

    @staticmethod
    def randn(shape, requires_grad: bool = False, seed: Optional[int] = None, name: Optional[str] = None):
        rng = Tensor._rng if seed is None else np.random.default_rng(seed)
        return Tensor(rng.standard_normal(size=shape, dtype=np.float32), requires_grad=requires_grad, name=name)

    def detach(self):
        return Tensor(self.data.copy(), requires_grad=False, name=self.name)

    def numpy(self) -> np.ndarray:
        return self.data

    def _set_ctx(self, parents: Sequence["Tensor"], backward: Callable[[np.ndarray], None]):
        self._ctx = _Context(parents=parents, backward=backward)

    def backward(self, grad: Optional[np.ndarray] = None):
        if not self.requires_grad:
            return
        if grad is None:
            grad = np.ones_like(self.data, dtype=np.float32)
        self.grad = grad if self.grad is None else self.grad + grad
        topo = self._topo_sort()
        for tensor in reversed(topo):
            if tensor._ctx is None:
                continue
            if tensor.grad is None:
                continue
            tensor._ctx.backward(tensor.grad)

    def _topo_sort(self) -> Sequence["Tensor"]:
        visited: Set[int] = set()
        topo: list[Tensor] = []

        def build(t: Tensor):
            if id(t) in visited:
                return
            visited.add(id(t))
            if t._ctx is not None:
                for p in t._ctx.parents:
                    build(p)
            topo.append(t)

        build(self)
        return topo

    def _ensure_tensor(self, other):
        return other if isinstance(other, Tensor) else Tensor(other)

    @staticmethod
    def _reduce_grad(grad: np.ndarray, shape) -> np.ndarray:
        grad_reduced = grad
        while grad_reduced.ndim > len(shape):
            grad_reduced = grad_reduced.sum(axis=0)
        for i, dim in enumerate(shape):
            if dim == 1:
                grad_reduced = grad_reduced.sum(axis=i, keepdims=True)
        return grad_reduced

    def __add__(self, other):
        other = self._ensure_tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

        def backward(grad):
            if self.requires_grad:
                grad_self = self._reduce_grad(grad, self.data.shape)
                self.grad = grad_self if self.grad is None else self.grad + grad_self
            if other.requires_grad:
                grad_other = self._reduce_grad(grad, other.data.shape)
                other.grad = grad_other if other.grad is None else other.grad + grad_other

        if out.requires_grad:
            out._set_ctx([self, other], backward)
        return out

    def __sub__(self, other):
        other = self._ensure_tensor(other)
        out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)

        def backward(grad):
            if self.requires_grad:
                grad_self = self._reduce_grad(grad, self.data.shape)
                self.grad = grad_self if self.grad is None else self.grad + grad_self
            if other.requires_grad:
                grad_other = self._reduce_grad(-grad, other.data.shape)
                other.grad = grad_other if other.grad is None else other.grad + grad_other

        if out.requires_grad:
            out._set_ctx([self, other], backward)
        return out

    def __mul__(self, other):
        other = self._ensure_tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)

        def backward(grad):
            if self.requires_grad:
                grad_self = self._reduce_grad(grad * other.data, self.data.shape)
                self.grad = grad_self if self.grad is None else self.grad + grad_self
            if other.requires_grad:
                grad_other = self._reduce_grad(grad * self.data, other.data.shape)
                other.grad = grad_other if other.grad is None else other.grad + grad_other

        if out.requires_grad:
            out._set_ctx([self, other], backward)
        return out

    def __truediv__(self, other):
        other = self._ensure_tensor(other)
        out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad)

        def backward(grad):
            if self.requires_grad:
                grad_self = self._reduce_grad(grad / other.data, self.data.shape)
                self.grad = grad_self if self.grad is None else self.grad + grad_self
            if other.requires_grad:
                grad_other = self._reduce_grad(-grad * self.data / (other.data ** 2), other.data.shape)
                other.grad = grad_other if other.grad is None else other.grad + grad_other

        if out.requires_grad:
            out._set_ctx([self, other], backward)
        return out

    def __pow__(self, power):
        out_data = self.data ** power
        out = Tensor(out_data, requires_grad=self.requires_grad)

        def backward(grad):
            if not self.requires_grad:
                return
            grad_self = grad * power * (self.data ** (power - 1))
            self.grad = grad_self if self.grad is None else self.grad + grad_self

        if out.requires_grad:
            out._set_ctx([self], backward)
        return out

    def __matmul__(self, other):
        other = self._ensure_tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)

        def backward(grad):
            grad_in = grad
            if grad_in.ndim == 1:
                grad_in = grad_in.reshape(1, -1)
            if self.requires_grad:
                grad_self = grad_in @ other.data.T
                self.grad = grad_self if self.grad is None else self.grad + grad_self
            if other.requires_grad:
                grad_self_data = np.atleast_2d(self.data)
                grad_in = np.atleast_2d(grad_in)
                if grad_in.shape[0] != grad_self_data.shape[0]:
                    if grad_in.shape[0] == 1:
                        grad_in = np.repeat(grad_in, grad_self_data.shape[0], axis=0)
                    elif grad_self_data.shape[0] == 1:
                        grad_self_data = np.repeat(grad_self_data, grad_in.shape[0], axis=0)
                    elif grad_in.shape[1] == grad_self_data.shape[0]:
                        grad_in = grad_in.T
                try:
                    grad_other = grad_self_data.T @ grad_in
                except ValueError:
                    grad_other = np.outer(grad_self_data.reshape(-1), grad_in.reshape(-1))
                other.grad = grad_other if other.grad is None else other.grad + grad_other

        if out.requires_grad:
            out._set_ctx([self, other], backward)
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)

        def backward(grad):
            if not self.requires_grad:
                return
            grad_self = grad
            if axis is not None and not keepdims:
                if isinstance(axis, tuple):
                    for ax in sorted(axis):
                        grad_self = np.expand_dims(grad_self, axis=ax)
                else:
                    grad_self = np.expand_dims(grad_self, axis=axis)
            grad_self = np.ones_like(self.data) * grad_self
            self.grad = grad_self if self.grad is None else self.grad + grad_self

        if out.requires_grad:
            out._set_ctx([self], backward)
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            denom = self.data.size
        elif isinstance(axis, tuple):
            denom = 1
            for ax in axis:
                denom *= self.data.shape[ax]
        else:
            denom = self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / denom

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), requires_grad=self.requires_grad)

        def backward(grad):
            if not self.requires_grad:
                return
            grad_self = grad * (self.data > 0)
            self.grad = grad_self if self.grad is None else self.grad + grad_self

        if out.requires_grad:
            out._set_ctx([self], backward)
        return out

    def tanh(self):
        out_data = np.tanh(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad)

        def backward(grad):
            if not self.requires_grad:
                return
            grad_self = grad * (1 - out_data ** 2)
            self.grad = grad_self if self.grad is None else self.grad + grad_self

        if out.requires_grad:
            out._set_ctx([self], backward)
        return out

    def softmax(self, axis=-1):
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        out_data = exp / exp.sum(axis=axis, keepdims=True)
        out = Tensor(out_data, requires_grad=self.requires_grad)

        def backward(grad):
            if not self.requires_grad:
                return
            grad_self = np.empty_like(self.data)
            for idx in np.ndindex(self.data.shape[:-1]):
                y = out_data[idx]
                jac = np.diag(y) - np.outer(y, y)
                grad_self[idx] = grad[idx] @ jac
            self.grad = grad_self if self.grad is None else self.grad + grad_self

        if out.requires_grad:
            out._set_ctx([self], backward)
        return out

    def log(self):
        out_data = np.log(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad)

        def backward(grad):
            if not self.requires_grad:
                return
            grad_self = grad / self.data
            self.grad = grad_self if self.grad is None else self.grad + grad_self

        if out.requires_grad:
            out._set_ctx([self], backward)
        return out

    def sqrt(self):
        out_data = np.sqrt(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad)

        def backward(grad):
            if not self.requires_grad:
                return
            grad_self = grad * 0.5 / (out_data + 1e-8)
            self.grad = grad_self if self.grad is None else self.grad + grad_self

        if out.requires_grad:
            out._set_ctx([self], backward)
        return out

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad}, name={self.name})"

    def slice(self, slices):
        out = Tensor(self.data[slices], requires_grad=self.requires_grad)

        def backward(grad):
            if not self.requires_grad:
                return
            grad_full = np.zeros_like(self.data)
            grad_full[slices] = grad
            self.grad = grad_full if self.grad is None else self.grad + grad_full

        if out.requires_grad:
            out._set_ctx([self], backward)
        return out


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    probs = logits.softmax(axis=-1)
    log_probs = probs.log()
    batch = targets.shape[0]
    losses = -log_probs.data[np.arange(batch), targets]
    loss_tensor = Tensor(losses.mean(), requires_grad=logits.requires_grad)

    def backward(grad):
        if not logits.requires_grad:
            return
        grad_logits = probs.data.copy()
        grad_logits[np.arange(batch), targets] -= 1
        grad_logits = grad_logits / batch
        logits.grad = grad * grad_logits if logits.grad is None else logits.grad + grad * grad_logits

    if loss_tensor.requires_grad:
        loss_tensor._set_ctx([logits], backward)
    return loss_tensor


def stack(tensors: Iterable[Tensor], axis=0) -> Tensor:
    data = np.stack([t.data for t in tensors], axis=axis)
    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(data, requires_grad=requires_grad)

    def backward(grad):
        for i, t in enumerate(tensors):
            if not t.requires_grad:
                continue
            grad_slice = np.take(grad, i, axis=axis)
            t.grad = grad_slice if t.grad is None else t.grad + grad_slice

    if out.requires_grad:
        out._set_ctx(list(tensors), backward)
    return out


def concatenate(tensors: Iterable[Tensor], axis=0) -> Tensor:
    data = np.concatenate([t.data for t in tensors], axis=axis)
    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(data, requires_grad=requires_grad)

    def backward(grad):
        splits = np.cumsum([t.data.shape[axis] for t in tensors])[:-1]
        grad_slices = np.split(grad, splits, axis=axis)
        for t, g in zip(tensors, grad_slices):
            if not t.requires_grad:
                continue
            t.grad = g if t.grad is None else t.grad + g

    if out.requires_grad:
        out._set_ctx(list(tensors), backward)
    return out
