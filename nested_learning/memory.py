"""Continuum Memory System (CMS) implementation using torch."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Sequence

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
    def __init__(
        self,
        features: int,
        frequency: int,
        depth: int,
        decay: float = 0.0,
        replay_ratio: float = 0.0,
        replay_steps: int = 1,
        replay_buffer: int = 128,
    ):
        super().__init__()
        self.frequency = frequency
        self.layers = nn.ModuleList([Linear(features, features) for _ in range(depth)])
        self.norm = LayerNorm(features)
        self.register_buffer("state_value", torch.zeros(1, features))
        self.state_time = 0
        self.decay = decay
        self.replay_ratio = replay_ratio
        self.replay_steps = replay_steps
        self.replay_buffer: Deque[torch.Tensor] = deque(maxlen=replay_buffer)
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
            if self.replay_ratio > 0.0 and self.replay_buffer:
                replay_count = max(1, int(self.replay_ratio * len(self.replay_buffer)))
                for _ in range(self.replay_steps):
                    replay_samples = list(self.replay_buffer)[-replay_count:]
                    replay_batch = torch.cat(replay_samples, dim=0)
                    replay_out = self.forward(replay_batch)
                    loss = loss + F.mse_loss(replay_out, replay_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step(lr_override=eta, weight_decay_override=alpha)
        with torch.no_grad():
            if self.decay > 0.0:
                self.state_value.mul_(1.0 - self.decay)
                self.state_value.add_(out.mean(dim=0, keepdim=True) * self.decay)
            else:
                self.state_value.copy_(out.mean(dim=0, keepdim=True))
            self.state_time = time
            self.replay_buffer.append(x_detached)

    def state(self) -> MemoryState:
        return MemoryState(time=self.state_time, value=self.state_value)

    def load_pretrained_layers(self, layers: Sequence[object]):
        if len(layers) != len(self.layers):
            raise ValueError("Pretrained layers count must match CMS block depth")
        for target, source in zip(self.layers, layers):
            if isinstance(source, dict):
                target.weight.data.copy_(source["weight"])
                if target.bias is not None and "bias" in source and source["bias"] is not None:
                    target.bias.data.copy_(source["bias"])
                continue
            if hasattr(source, "weight"):
                target.weight.data.copy_(source.weight.data)
                if target.bias is not None and getattr(source, "bias", None) is not None:
                    target.bias.data.copy_(source.bias.data)
                continue
            raise ValueError("Unsupported pretrained layer type")


class ContinuumMemorySystem(nn.Module):
    """Multi-timescale memory system as a chain of memory blocks."""

    def __init__(
        self,
        features: int,
        frequencies: List[int],
        depth: int = 2,
        decay: float | Sequence[float] = 0.0,
        replay_ratio: float | Sequence[float] = 0.0,
        replay_steps: int = 1,
        replay_buffer: int = 128,
    ):
        super().__init__()
        decay_list = list(decay) if isinstance(decay, (list, tuple)) else [decay] * len(frequencies)
        replay_list = list(replay_ratio) if isinstance(replay_ratio, (list, tuple)) else [replay_ratio] * len(frequencies)
        self.blocks = nn.ModuleList(
            [
                MemoryBlock(
                    features,
                    freq,
                    depth,
                    decay=decay_list[idx],
                    replay_ratio=replay_list[idx],
                    replay_steps=replay_steps,
                    replay_buffer=replay_buffer,
                )
                for idx, freq in enumerate(frequencies)
            ]
        )
        self.features = features

    def forward(self, x: torch.Tensor, time: int, update: bool = True) -> torch.Tensor:
        context = x
        for block in self.blocks:
            block.update(context, time, update=update)
            context = context + block.state_value
        return context

    def states(self) -> List[MemoryState]:
        return [block.state() for block in self.blocks]

    def load_pretrained_blocks(self, blocks: Sequence[Sequence[object]]):
        if len(blocks) != len(self.blocks):
            raise ValueError("Pretrained blocks count must match CMS blocks")
        for block, layers in zip(self.blocks, blocks):
            block.load_pretrained_layers(layers)


class NestedContinuumMemorySystem(nn.Module):
    """Fully nested CMS where each block owns a sub-CMS."""

    def __init__(
        self,
        features: int,
        frequencies: List[int],
        depth: int = 2,
        decay: float | Sequence[float] = 0.0,
        replay_ratio: float | Sequence[float] = 0.0,
        replay_steps: int = 1,
        replay_buffer: int = 128,
    ):
        super().__init__()
        decay_list = list(decay) if isinstance(decay, (list, tuple)) else [decay] * len(frequencies)
        replay_list = list(replay_ratio) if isinstance(replay_ratio, (list, tuple)) else [replay_ratio] * len(frequencies)
        self.blocks = nn.ModuleList(
            [
                MemoryBlock(
                    features,
                    freq,
                    depth,
                    decay=decay_list[idx],
                    replay_ratio=replay_list[idx],
                    replay_steps=replay_steps,
                    replay_buffer=replay_buffer,
                )
                for idx, freq in enumerate(frequencies)
            ]
        )
        self.subsystems = nn.ModuleList(
            [
                ContinuumMemorySystem(
                    features,
                    frequencies[: idx + 1],
                    depth=depth,
                    decay=decay_list[: idx + 1],
                    replay_ratio=replay_list[: idx + 1],
                    replay_steps=replay_steps,
                    replay_buffer=replay_buffer,
                )
                for idx in range(len(frequencies))
            ]
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

    def load_pretrained_blocks(self, blocks: Sequence[Sequence[object]]):
        if len(blocks) != len(self.blocks):
            raise ValueError("Pretrained blocks count must match CMS blocks")
        for idx, (block, layers) in enumerate(zip(self.blocks, blocks)):
            block.load_pretrained_layers(layers)
            self.subsystems[idx].load_pretrained_blocks(blocks[: idx + 1])


class SequentialContinuumMemorySystem(nn.Module):
    """Sequential CMS variant with explicit pass-through normalization."""

    def __init__(
        self,
        features: int,
        frequencies: List[int],
        depth: int = 2,
        decay: float | Sequence[float] = 0.0,
        replay_ratio: float | Sequence[float] = 0.0,
        replay_steps: int = 1,
        replay_buffer: int = 128,
    ):
        super().__init__()
        decay_list = list(decay) if isinstance(decay, (list, tuple)) else [decay] * len(frequencies)
        replay_list = list(replay_ratio) if isinstance(replay_ratio, (list, tuple)) else [replay_ratio] * len(frequencies)
        self.blocks = nn.ModuleList(
            [
                MemoryBlock(
                    features,
                    freq,
                    depth,
                    decay=decay_list[idx],
                    replay_ratio=replay_list[idx],
                    replay_steps=replay_steps,
                    replay_buffer=replay_buffer,
                )
                for idx, freq in enumerate(frequencies)
            ]
        )
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

    def load_pretrained_blocks(self, blocks: Sequence[Sequence[object]]):
        if len(blocks) != len(self.blocks):
            raise ValueError("Pretrained blocks count must match CMS blocks")
        for block, layers in zip(self.blocks, blocks):
            block.load_pretrained_layers(layers)


class HeadwiseContinuumMemorySystem(nn.Module):
    """Independent head-wise CMS for parallel memory streams."""

    def __init__(
        self,
        features: int,
        frequencies: List[int],
        heads: int = 4,
        depth: int = 2,
        decay: float | Sequence[float] = 0.0,
        replay_ratio: float | Sequence[float] = 0.0,
        replay_steps: int = 1,
        replay_buffer: int = 128,
    ):
        super().__init__()
        if features % heads != 0:
            raise ValueError("features must be divisible by heads")
        self.heads = heads
        self.head_dim = features // heads
        self.systems = nn.ModuleList(
            [
                ContinuumMemorySystem(
                    self.head_dim,
                    frequencies,
                    depth=depth,
                    decay=decay,
                    replay_ratio=replay_ratio,
                    replay_steps=replay_steps,
                    replay_buffer=replay_buffer,
                )
                for _ in range(heads)
            ]
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

    def load_pretrained_blocks(self, blocks: Sequence[Sequence[object]]):
        for system in self.systems:
            system.load_pretrained_blocks(blocks)
