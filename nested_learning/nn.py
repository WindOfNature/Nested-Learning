"""Torch-based neural modules with Nested Learning inspired updates."""

from __future__ import annotations

from typing import Iterable, List, Optional

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
        if self.use_kernels and not torch.is_grad_enabled() and not x.is_cuda:
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


class SelfModifyingLayer(nn.Module):
    """Layer with learned update rule for self-modification."""

    def __init__(self, features: int, update_hidden: int = 64):
        super().__init__()
        self.base = Linear(features, features)
        self.rule = MLP(features * 2, update_hidden, features)
        self.scale = nn.Parameter(torch.ones(features))
        self.base_optimizer = DGD(self.base.parameters(), lr=1e-3, beta=0.9, alpha=0.5, weight_decay=1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.base(x))

    def self_update(self, x: torch.Tensor, grad: torch.Tensor):
        x_detached = x.detach()
        grad_detached = grad.detach()
        context = torch.cat([x_detached, grad_detached], dim=-1)
        delta = torch.tanh(self.rule(context))
        target = x_detached + grad_detached
        pred = self.base(x_detached)
        loss = F.mse_loss(pred, target)
        self.base_optimizer.zero_grad()
        loss.backward()
        self.base_optimizer.step()
        with torch.no_grad():
            self.base.weight.add_(self.scale * delta.mean(dim=0))


class SelfReferentialTitan(nn.Module):
    """Self-referential module with an inner optimizer for its update rule."""

    def __init__(self, features: int, update_hidden: int = 64, inner_lr: float = 1e-3):
        super().__init__()
        self.core = SelfModifyingLayer(features, update_hidden=update_hidden)
        self.rule_optimizer = torch.optim.AdamW(self.core.rule.parameters(), lr=inner_lr, weight_decay=1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)

    def self_update(self, x: torch.Tensor, grad: torch.Tensor):
        x_detached = x.detach()
        grad_detached = grad.detach()
        self.core.self_update(x_detached, grad_detached)
        target = -grad_detached
        context = torch.cat([x_detached, grad_detached], dim=-1)
        pred = self.core.rule(context)
        loss = F.mse_loss(pred, target)
        self.rule_optimizer.zero_grad()
        loss.backward()
        self.rule_optimizer.step()


class SelfModifyingStack(nn.Module):
    """Stacked self-modifying titans for deeper self-referential updates."""

    def __init__(self, features: int, depth: int = 2, update_hidden: int = 64, inner_lr: float = 1e-3):
        super().__init__()
        self.layers = nn.ModuleList(
            [SelfReferentialTitan(features, update_hidden=update_hidden, inner_lr=inner_lr) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = F.relu(layer(out))
        return out

    def self_update(self, x: torch.Tensor, grad: torch.Tensor):
        for layer in self.layers:
            layer.self_update(x, grad)
