"""Continuum Memory System (CMS) implementation using torch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import Linear, LayerNorm, MemoryMLP
from .torch_optim import DGD


@dataclass
class MemoryState:
    time: int
    value: torch.Tensor


class MemoryBlock(nn.Module):
    def __init__(self, features: int, frequency: int, depth: int):
        super().__init__()
        self.frequency = frequency
        self.layers = nn.ModuleList([Linear(features, features) for _ in range(depth)])
        self.norm = LayerNorm(features)
        self.register_buffer("state_value", torch.zeros(1, features))
        self.state_time = 0
        self.optimizer = DGD(self.parameters(), lr=1e-3, beta=0.9, alpha=0.5, weight_decay=1e-4)
        self.meta = MemoryMLP(features)
        self.malpha = MemoryMLP(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = F.relu(layer(out))
        return self.norm(out)

    def update(self, x: torch.Tensor, time: int, update: bool = True):
        if not update or time % self.frequency != 0:
            return
        with torch.enable_grad():
            x_detached = x.detach()
            out = self.forward(x_detached)
            eta = F.softplus(self.meta(x_detached)).mean().clamp(min=1e-5).item()
            alpha = torch.sigmoid(self.malpha(x_detached)).mean().clamp(min=1e-5).item()
            loss = F.mse_loss(out, x_detached)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step(lr_override=eta, weight_decay_override=alpha)
        with torch.no_grad():
            self.state_value.copy_(out.mean(dim=0, keepdim=True))
            self.state_time = time

    def state(self) -> MemoryState:
        return MemoryState(time=self.state_time, value=self.state_value)


class ContinuumMemorySystem(nn.Module):
    """Multi-timescale memory system as a chain of memory blocks."""

    def __init__(self, features: int, frequencies: List[int], depth: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([MemoryBlock(features, freq, depth) for freq in frequencies])
        self.features = features

    def forward(self, x: torch.Tensor, time: int, update: bool = True) -> torch.Tensor:
        context = x
        for block in self.blocks:
            block.update(context, time, update=update)
            context = context + block.state_value
        return context

    def states(self) -> List[MemoryState]:
        return [block.state() for block in self.blocks]


class NestedContinuumMemorySystem(nn.Module):
    """Fully nested CMS where each block owns a sub-CMS."""

    def __init__(self, features: int, frequencies: List[int], depth: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([MemoryBlock(features, freq, depth) for freq in frequencies])
        self.subsystems = nn.ModuleList(
            [ContinuumMemorySystem(features, frequencies[: idx + 1], depth=depth) for idx in range(len(frequencies))]
        )
        self.features = features

    def forward(self, x: torch.Tensor, time: int, update: bool = True) -> torch.Tensor:
        context = x
        for block, sub in zip(self.blocks, self.subsystems):
            block.update(context, time, update=update)
            nested_context = sub.forward(context, time, update=update)
            context = context + block.state_value + nested_context
        return context

    def states(self) -> List[MemoryState]:
        states: List[MemoryState] = []
        for block, sub in zip(self.blocks, self.subsystems):
            states.append(block.state())
            states.extend(sub.states())
        return states


class SequentialContinuumMemorySystem(nn.Module):
    """Sequential CMS variant with explicit pass-through normalization."""

    def __init__(self, features: int, frequencies: List[int], depth: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([MemoryBlock(features, freq, depth) for freq in frequencies])
        self.norm = LayerNorm(features)
        self.features = features

    def forward(self, x: torch.Tensor, time: int, update: bool = True) -> torch.Tensor:
        context = x
        for block in self.blocks:
            block.update(context, time, update=update)
            context = self.norm(context + block.state_value)
        return context

    def states(self) -> List[MemoryState]:
        return [block.state() for block in self.blocks]


class HeadwiseContinuumMemorySystem(nn.Module):
    """Independent head-wise CMS for parallel memory streams."""

    def __init__(self, features: int, frequencies: List[int], heads: int = 4, depth: int = 2):
        super().__init__()
        if features % heads != 0:
            raise ValueError("features must be divisible by heads")
        self.heads = heads
        self.head_dim = features // heads
        self.systems = nn.ModuleList(
            [ContinuumMemorySystem(self.head_dim, frequencies, depth=depth) for _ in range(heads)]
        )
        self.features = features

    def forward(self, x: torch.Tensor, time: int, update: bool = True) -> torch.Tensor:
        chunks = []
        for head in range(self.heads):
            start = head * self.head_dim
            end = start + self.head_dim
            chunk = x[:, start:end]
            chunks.append(self.systems[head].forward(chunk, time, update=update))
        return torch.cat(chunks, dim=-1)

    def states(self) -> List[MemoryState]:
        states: List[MemoryState] = []
        for system in self.systems:
            states.extend(system.states())
        return states
