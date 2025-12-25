"""Torch-based neural modules with Nested Learning inspired updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernels import cpu as cpu_kernels
from .kernels import gpu as gpu_kernels
from .torch_optim import DGD


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, use_kernels: bool = True):
        super().__init__()
        scale = (2.0 / in_features) ** 0.5
        weight = torch.randn(in_features, out_features) * scale
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.use_kernels = use_kernels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if self.use_kernels and not torch.is_grad_enabled():
            if x.is_cuda and gpu_kernels.available().available:
                out = gpu_kernels.matmul(x, self.weight)
            elif not x.is_cuda:
                out = cpu_kernels.matmul_torch(x, self.weight)
            else:
                out = x @ self.weight
        else:
            out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-5, use_kernels: bool = True):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps
        self.use_kernels = use_kernels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_kernels and not torch.is_grad_enabled():
            if x.is_cuda and gpu_kernels.available().available:
                return gpu_kernels.layernorm(x, self.gamma, self.beta, self.eps)
            if not x.is_cuda:
                return cpu_kernels.layernorm_torch(x, self.gamma, self.beta, self.eps)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.fc1 = Linear(in_features, hidden_features)
        self.fc2 = Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class MemoryMLP(nn.Module):
    """Memory module M□(·) = (·) + W1 σ(W2 (·))."""

    def __init__(self, features: int, hidden_features: int | None = None):
        super().__init__()
        hidden = hidden_features or features * 2
        self.fc1 = Linear(features, hidden)
        self.fc2 = Linear(hidden, features)
        self.optimizer = DGD(self.parameters(), lr=1e-3, beta=0.9, alpha=0.5, weight_decay=1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.relu(self.fc1(x)))

    def dgd_update(self, keys: torch.Tensor, targets: torch.Tensor, lr: float, weight_decay: float):
        with torch.enable_grad():
            pred = self.forward(keys)
            loss = F.mse_loss(pred, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step(lr_override=lr, weight_decay_override=weight_decay)


class MemoryMatrix(nn.Module):
    """Associative memory matrix M□ with residual connection."""

    def __init__(self, features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(features, features) * (1.0 / features**0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x @ self.weight

    def update(self, keys: torch.Tensor, targets: torch.Tensor, eta: float, alpha: float):
        with torch.enable_grad():
            pred = self.forward(keys)
            loss = F.mse_loss(pred, targets)
            loss.backward()
        with torch.no_grad():
            self.weight.mul_(alpha)
            if self.weight.grad is not None:
                self.weight.add_(self.weight.grad, alpha=-eta)
            kk = (keys.t() @ keys) / max(keys.shape[0], 1)
            self.weight.add_(kk, alpha=-eta)
            self.weight.grad = None


@dataclass
class MemorySignals:
    k: torch.Tensor
    v: torch.Tensor
    q: torch.Tensor
    eta: torch.Tensor
    alpha: torch.Tensor
    memory: torch.Tensor


class SelfReferentialTitan(nn.Module):
    """Self-referential module with explicit memories for {k, v, q, eta, alpha, memory}."""

    def __init__(self, features: int, update_hidden: int = 64):
        super().__init__()
        self.mk = MemoryMatrix(features)
        self.mv = MemoryMatrix(features)
        self.mq = MemoryMatrix(features)
        self.meta = MemoryMatrix(features)
        self.malpha = MemoryMatrix(features)
        self.mmemory = MemoryMatrix(features)
        self.scale = nn.Parameter(torch.ones(features))

    @staticmethod
    def _l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signals = self.generate_signals(x)
        return x + signals.memory

    def generate_signals(self, x: torch.Tensor) -> MemorySignals:
        q = self._l2_normalize(self.mq(x))
        k = self._l2_normalize(self.mk(x))
        v = self.mv(x)
        eta = F.softplus(self.meta(x))
        alpha = torch.sigmoid(self.malpha(x))
        memory = self.mmemory(q)
        return MemorySignals(k=k, v=v, q=q, eta=eta, alpha=alpha, memory=memory)

    def update_chunk(self, x: torch.Tensor):
        x_detached = x.detach()
        signals = self.generate_signals(x_detached)
        eta = signals.eta.mean().clamp(min=1e-5).item()
        alpha = signals.alpha.mean().clamp(min=1e-5).item()

        v_hat_k = self.mk(signals.v.detach())
        v_hat_v = self.mv(signals.v.detach())
        v_hat_q = self.mq(signals.v.detach())
        v_hat_eta = self.meta(signals.v.detach())
        v_hat_alpha = self.malpha(signals.v.detach())
        v_hat_memory = self.mmemory(signals.v.detach())

        keys = signals.k.detach()
        self.mk.update(keys, v_hat_k.detach(), eta=eta, alpha=alpha)
        self.mv.update(keys, v_hat_v.detach(), eta=eta, alpha=alpha)
        self.mq.update(keys, v_hat_q.detach(), eta=eta, alpha=alpha)
        self.meta.update(keys, v_hat_eta.detach(), eta=eta, alpha=alpha)
        self.malpha.update(keys, v_hat_alpha.detach(), eta=eta, alpha=alpha)
        self.mmemory.update(keys, v_hat_memory.detach(), eta=eta, alpha=alpha)


class SelfModifyingStack(nn.Module):
    """Stacked self-referential titans for deeper self-modifying updates."""

    def __init__(self, features: int, depth: int = 2, update_hidden: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([SelfReferentialTitan(features, update_hidden=update_hidden) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = F.relu(layer(out))
        return out

    def update_chunk(self, x: torch.Tensor):
        for layer in self.layers:
            layer.update_chunk(x)
