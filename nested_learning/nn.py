"""Torch-based neural modules with Nested Learning inspired updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

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


@dataclass
class AssociativeMemoryState:
    memory: torch.Tensor
    time: int


class AssociativeMemory(nn.Module):
    """Associative memory with parametric and non-parametric modes."""

    def __init__(
        self,
        features: int,
        value_dim: Optional[int] = None,
        mode: Literal["parametric", "nonparametric"] = "parametric",
        rule: Literal["hebbian", "delta", "oja"] = "delta",
    ):
        super().__init__()
        self.features = features
        self.value_dim = value_dim or features
        self.mode = mode
        self.rule = rule
        self.register_buffer("memory", torch.zeros(self.value_dim, self.features))
        self.state = AssociativeMemoryState(memory=self.memory, time=0)
        if mode == "parametric":
            self.memory_net = MemoryMLP(features, hidden_features=features * 2)
        else:
            self.memory_net = None

    def forward(self, keys: torch.Tensor, values: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        if self.mode == "nonparametric":
            scores = queries @ keys.t()
            weights = torch.softmax(scores, dim=-1)
            return weights @ values
        if self.memory_net is None:
            raise RuntimeError("Parametric memory net not initialized")
        return self.memory_net(queries)

    def update_memory(self, keys: torch.Tensor, values: torch.Tensor, eta: float, alpha: float):
        if self.mode == "parametric":
            if self.memory_net is None:
                return
            self.memory_net.dgd_update(keys, values, lr=eta, weight_decay=alpha)
            return
        with torch.no_grad():
            for k, v in zip(keys, values):
                k = k.unsqueeze(-1)
                v = v.unsqueeze(-1)
                if self.rule == "hebbian":
                    self.memory.mul_(alpha).add_(v @ k.t(), alpha=eta)
                elif self.rule == "oja":
                    self.memory.mul_(alpha).add_(v @ k.t(), alpha=eta)
                    self.memory.sub_(self.memory.t() @ v @ k.t(), alpha=eta)
                else:  # delta rule
                    pred = self.memory @ k
                    self.memory.mul_(alpha).add_((v - pred) @ k.t(), alpha=eta)
            self.state = AssociativeMemoryState(memory=self.memory, time=self.state.time + 1)


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

    def __init__(self, features: int, update_hidden: int = 64, query_static: bool = False):
        super().__init__()
        self.mk = MemoryMLP(features, hidden_features=update_hidden)
        self.mv = MemoryMLP(features, hidden_features=update_hidden)
        self.mq = MemoryMLP(features, hidden_features=update_hidden)
        self.meta = MemoryMLP(features, hidden_features=update_hidden)
        self.malpha = MemoryMLP(features, hidden_features=update_hidden)
        self.mmemory = MemoryMLP(features, hidden_features=update_hidden)
        self.query_static = query_static
        self.q_proj = Linear(features, features) if query_static else None
        self.scale = nn.Parameter(torch.ones(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signals = self.generate_signals(x)
        return x + signals.memory

    def generate_signals(self, x: torch.Tensor) -> MemorySignals:
        q = self.q_proj(x) if self.q_proj is not None else self.mq(x)
        k = self.mk(x)
        v = self.mv(x)
        eta = F.softplus(self.meta(x))
        alpha = torch.sigmoid(self.malpha(x))
        memory = self.mmemory(q)
        return MemorySignals(k=k, v=v, q=q, eta=eta, alpha=alpha, memory=memory)

    def _update_modules(
        self,
        signals: MemorySignals,
        update_memory: bool = True,
        update_projections: bool = True,
    ):
        eta = signals.eta.mean().clamp(min=1e-5).item()
        alpha = signals.alpha.mean().clamp(min=1e-5).item()

        v_hat = signals.v.detach()
        if update_projections:
            v_hat_k = self.mk(v_hat)
            v_hat_v = self.mv(v_hat)
            v_hat_eta = self.meta(v_hat)
            v_hat_alpha = self.malpha(v_hat)
            self.mk.dgd_update(signals.k.detach(), v_hat_k.detach(), lr=eta, weight_decay=alpha)
            self.mv.dgd_update(signals.k.detach(), v_hat_v.detach(), lr=eta, weight_decay=alpha)
            if self.q_proj is None:
                v_hat_q = self.mq(v_hat)
                self.mq.dgd_update(signals.k.detach(), v_hat_q.detach(), lr=eta, weight_decay=alpha)
            self.meta.dgd_update(signals.k.detach(), v_hat_eta.detach(), lr=eta, weight_decay=alpha)
            self.malpha.dgd_update(signals.k.detach(), v_hat_alpha.detach(), lr=eta, weight_decay=alpha)
        if update_memory:
            v_hat_memory = self.mmemory(v_hat)
            self.mmemory.dgd_update(signals.k.detach(), v_hat_memory.detach(), lr=eta, weight_decay=alpha)
            with torch.no_grad():
                self.mmemory.fc2.weight.add_(self.scale * signals.memory.mean(dim=0))

    def _compute_signals(self, x: torch.Tensor) -> MemorySignals:
        x_detached = x.detach()
        return self.generate_signals(x_detached)

    def update_chunk(self, x: torch.Tensor, chunk_size: int | None = None, memory_chunk_size: int | None = None):
        chunk_size = chunk_size or x.shape[0]
        memory_chunk_size = memory_chunk_size or chunk_size
        if chunk_size <= 0 or memory_chunk_size <= 0:
            raise ValueError("chunk_size and memory_chunk_size must be positive")
        for start in range(0, x.shape[0], chunk_size):
            chunk = x[start : start + chunk_size]
            signals = self._compute_signals(chunk)
            self._update_modules(signals, update_memory=False, update_projections=True)
        for start in range(0, x.shape[0], memory_chunk_size):
            chunk = x[start : start + memory_chunk_size]
            signals = self._compute_signals(chunk)
            self._update_modules(signals, update_memory=True, update_projections=False)


class SelfModifyingStack(nn.Module):
    """Stacked self-referential titans for deeper self-modifying updates."""

    def __init__(
        self,
        features: int,
        depth: int = 2,
        update_hidden: int = 64,
        query_static: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SelfReferentialTitan(features, update_hidden=update_hidden, query_static=query_static)
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = F.relu(layer(out))
        return out

    def update_chunk(self, x: torch.Tensor, chunk_size: int | None = None, memory_chunk_size: int | None = None):
        for layer in self.layers:
            layer.update_chunk(x, chunk_size=chunk_size, memory_chunk_size=memory_chunk_size)


@dataclass
class ContextFlowState:
    time: int
    level: int
    context: torch.Tensor


class ContextFlowLevel(nn.Module):
    """Single level of nested context flow with its own optimization dynamics."""

    def __init__(self, features: int, hidden: int = 128):
        super().__init__()
        self.transform = nn.Sequential(
            Linear(features, hidden),
            nn.ReLU(),
            Linear(hidden, features),
        )
        self.norm = LayerNorm(features)
        self.meta = MemoryMLP(features)
        self.malpha = MemoryMLP(features)
        self.optimizer = DGD(self.parameters(), lr=1e-3, beta=0.9, alpha=0.5, weight_decay=1e-4)
        self.state = ContextFlowState(time=0, level=0, context=torch.zeros(1, features))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.norm(context + self.transform(context))

    def update(self, context: torch.Tensor, time: int):
        with torch.enable_grad():
            context_detached = context.detach()
            eta = F.softplus(self.meta(context_detached)).mean().clamp(min=1e-5).item()
            alpha = torch.sigmoid(self.malpha(context_detached)).mean().clamp(min=1e-5).item()
            output = self.forward(context_detached)
            target = context_detached + self.transform(context_detached).detach()
            loss = F.mse_loss(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step(lr_override=eta, weight_decay_override=alpha)
        with torch.no_grad():
            self.state = ContextFlowState(time=time, level=self.state.level, context=output.mean(dim=0, keepdim=True))


class NestedContextFlow(nn.Module):
    """Nested multi-level context flow module."""

    def __init__(self, features: int, depth: int = 2, hidden: int = 128):
        super().__init__()
        self.levels = nn.ModuleList([ContextFlowLevel(features, hidden=hidden) for _ in range(depth)])

    def forward(self, context: torch.Tensor, time: int, update: bool = True) -> torch.Tensor:
        flow = context
        for idx, level in enumerate(self.levels):
            if update:
                level.state = ContextFlowState(time=time, level=idx, context=flow.detach())
                level.update(flow, time)
            flow = level.forward(flow)
        return flow
