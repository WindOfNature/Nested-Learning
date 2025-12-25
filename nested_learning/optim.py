"""Optimizer suite inspired by nested learning paper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np

from .tensor import Tensor


class Optimizer:
    def __init__(self, params: Iterable[Tensor], lr: float = 1e-3):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params: Iterable[Tensor], lr: float = 1e-2, momentum: float = 0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for idx, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.velocity[idx] = self.momentum * self.velocity[idx] + p.grad
            p.data -= self.lr * self.velocity[idx]


class Adam(Optimizer):
    def __init__(self, params: Iterable[Tensor], lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for idx, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * p.grad
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (p.grad ** 2)
            m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
            v_hat = self.v[idx] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    def __init__(self, params: Iterable[Tensor], lr: float = 1e-3, weight_decay: float = 1e-2, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        self.weight_decay = weight_decay

    def step(self):
        self.t += 1
        for idx, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * p.grad
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (p.grad ** 2)
            m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
            v_hat = self.v[idx] / (1 - self.beta2 ** self.t)
            p.data *= (1 - self.lr * self.weight_decay)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


@dataclass
class ExpressiveState:
    memory: np.ndarray
    covariance: np.ndarray


class ExpressiveOptimizer(Optimizer):
    """Associative-memory optimizer based on NL objective formulations."""

    def __init__(self, params: Iterable[Tensor], lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.99, lam: float = 1e-4):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.lam = lam
        self.states: List[ExpressiveState] = []
        for p in self.params:
            self.states.append(
                ExpressiveState(
                    memory=np.zeros_like(p.data),
                    covariance=np.zeros_like(p.data),
                )
            )

    def step(self):
        for p, state in zip(self.params, self.states):
            if p.grad is None:
                continue
            state.memory = self.beta1 * state.memory + (1 - self.beta1) * p.grad
            state.covariance = self.beta2 * state.covariance + (1 - self.beta2) * (p.grad ** 2)
            denom = np.sqrt(state.covariance) + self.lam
            update = state.memory / denom
            p.data -= self.lr * update


class Muon(Optimizer):
    """Muon optimizer: expressive multi-step momentum with orthogonal stabilization."""

    def __init__(self, params: Iterable[Tensor], lr: float = 1e-3, momentum: float = 0.95, beta: float = 0.5, eps: float = 1e-8):
        super().__init__(params, lr)
        self.momentum = momentum
        self.beta = beta
        self.eps = eps
        self.velocity = [np.zeros_like(p.data) for p in self.params]
        self.energy = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for idx, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.velocity[idx] = self.momentum * self.velocity[idx] + (1 - self.momentum) * p.grad
            self.energy[idx] = self.beta * self.energy[idx] + (1 - self.beta) * (p.grad ** 2)
            denom = np.sqrt(self.energy[idx]) + self.eps
            update = self.velocity[idx] / denom
            p.data -= self.lr * update


class ShampooOptimizer(Optimizer):
    """Outer-product associative memory optimizer (Shampoo-like)."""

    def __init__(self, params: Iterable[Tensor], lr: float = 1e-3, beta2: float = 0.95, eps: float = 1e-4):
        super().__init__(params, lr)
        self.beta2 = beta2
        self.eps = eps
        self.left_stats = []
        self.right_stats = []
        for p in self.params:
            if p.data.ndim == 2:
                m, n = p.data.shape
                self.left_stats.append(np.eye(m, dtype=np.float32))
                self.right_stats.append(np.eye(n, dtype=np.float32))
            else:
                self.left_stats.append(np.zeros_like(p.data))
                self.right_stats.append(None)

    def _inv_root(self, matrix: np.ndarray) -> np.ndarray:
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.maximum(eigvals, self.eps)
        inv_root = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        return inv_root

    def step(self):
        for idx, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad
            if grad.ndim != 2:
                p.data -= self.lr * grad
                continue
            left = self.left_stats[idx]
            right = self.right_stats[idx]
            left = self.beta2 * left + (1 - self.beta2) * (grad @ grad.T)
            right = self.beta2 * right + (1 - self.beta2) * (grad.T @ grad)
            self.left_stats[idx] = left
            self.right_stats[idx] = right
            left_inv = self._inv_root(left + self.eps * np.eye(left.shape[0]))
            right_inv = self._inv_root(right + self.eps * np.eye(right.shape[0]))
            precond_grad = left_inv @ grad @ right_inv
            p.data -= self.lr * precond_grad
