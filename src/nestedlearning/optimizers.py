"""Custom optimizers without torch dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from nestedlearning.tensor import SimpleTensor


@dataclass
class Optimizer:
    params: Iterable[SimpleTensor]
    lr: float = 1e-3

    def step(self) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()


@dataclass
class Adam(Optimizer):
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    t: int = 0
    m: list[np.ndarray] = field(default_factory=list)
    v: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        for idx, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * p.grad
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (p.grad ** 2)
            m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
            v_hat = self.v[idx] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


@dataclass
class AdamW(Adam):
    weight_decay: float = 0.01

    def step(self) -> None:
        self.t += 1
        for idx, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * p.grad
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (p.grad ** 2)
            m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
            v_hat = self.v[idx] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * p.data)


@dataclass
class Muon(Optimizer):
    """Placeholder for the Muon optimizer described in the paper."""

    def step(self) -> None:
        raise NotImplementedError("Muon optimizer is not implemented yet.")


@dataclass
class DeltaGradientDescent(Optimizer):
    """Delta Gradient Descent with optional normalization.

    Expects a single weight matrix and context (input, grad_output) to update.
    """

    alpha: float = 1e-3
    beta: float = 1e-3
    eps: float = 1e-8

    def step(self) -> None:
        raise NotImplementedError("Use step_with_context for DeltaGradientDescent.")

    def step_with_context(self, param: SimpleTensor, x: np.ndarray, grad_y: np.ndarray) -> None:
        if param.data.ndim != 2:
            raise ValueError("DeltaGradientDescent expects a 2D weight matrix.")
        x_vec = x.reshape(-1, 1).astype(np.float32)
        grad_y_vec = grad_y.reshape(-1, 1).astype(np.float32)
        norm = float(x_vec.T @ x_vec) + self.eps
        alpha = self.alpha / norm
        identity = np.eye(x_vec.shape[0], dtype=np.float32)
        param.data = param.data @ (identity - alpha * (x_vec @ x_vec.T)) - self.beta * (grad_y_vec @ x_vec.T)
