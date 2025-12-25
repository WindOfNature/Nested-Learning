"""Continuum Memory System (CMS) implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .tensor import Tensor, concatenate
from .nn import Linear, LayerNorm


@dataclass
class MemoryState:
    time: int
    value: Tensor


class MemoryBlock:
    def __init__(self, features: int, frequency: int, depth: int):
        self.frequency = frequency
        self.layers = [Linear(features, features) for _ in range(depth)]
        self.norm = LayerNorm(features)
        self.state = MemoryState(time=0, value=Tensor.zeros((1, features)))

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            out = layer(out).relu()
        return self.norm(out)

    def update(self, x: Tensor, time: int):
        if time % self.frequency != 0:
            return
        memory_value = self.forward(x).mean(axis=0, keepdims=True)
        self.state = MemoryState(time=time, value=memory_value)


class ContinuumMemorySystem:
    """Multi-timescale memory system as a chain of memory blocks."""

    def __init__(self, features: int, frequencies: List[int], depth: int = 2):
        self.blocks = [MemoryBlock(features, freq, depth) for freq in frequencies]
        self.features = features

    def forward(self, x: Tensor, time: int) -> Tensor:
        context = x
        for block in self.blocks:
            block.update(context, time)
            context = context + block.state.value
        return context

    def states(self) -> List[MemoryState]:
        return [block.state for block in self.blocks]


class NestedContinuumMemorySystem:
    """Fully nested CMS where each block owns a sub-CMS."""

    def __init__(self, features: int, frequencies: List[int], depth: int = 2):
        self.blocks = [MemoryBlock(features, freq, depth) for freq in frequencies]
        self.subsystems = [ContinuumMemorySystem(features, frequencies[: idx + 1], depth=depth) for idx in range(len(frequencies))]
        self.features = features

    def forward(self, x: Tensor, time: int) -> Tensor:
        context = x
        for block, sub in zip(self.blocks, self.subsystems):
            block.update(context, time)
            nested_context = sub.forward(context, time)
            context = context + block.state.value + nested_context
        return context

    def states(self) -> List[MemoryState]:
        states: List[MemoryState] = []
        for block, sub in zip(self.blocks, self.subsystems):
            states.append(block.state)
            states.extend(sub.states())
        return states


class SequentialContinuumMemorySystem:
    """Sequential CMS variant with explicit pass-through normalization."""

    def __init__(self, features: int, frequencies: List[int], depth: int = 2):
        self.blocks = [MemoryBlock(features, freq, depth) for freq in frequencies]
        self.norm = LayerNorm(features)
        self.features = features

    def forward(self, x: Tensor, time: int) -> Tensor:
        context = x
        for block in self.blocks:
            block.update(context, time)
            context = self.norm(context + block.state.value)
        return context

    def states(self) -> List[MemoryState]:
        return [block.state for block in self.blocks]


class HeadwiseContinuumMemorySystem:
    """Independent head-wise CMS for parallel memory streams."""

    def __init__(self, features: int, frequencies: List[int], heads: int = 4, depth: int = 2):
        if features % heads != 0:
            raise ValueError("features must be divisible by heads")
        self.heads = heads
        self.head_dim = features // heads
        self.systems = [ContinuumMemorySystem(self.head_dim, frequencies, depth=depth) for _ in range(heads)]
        self.features = features

    def forward(self, x: Tensor, time: int) -> Tensor:
        chunks = []
        for head in range(self.heads):
            start = head * self.head_dim
            end = start + self.head_dim
            chunk = x.slice((slice(None), slice(start, end)))
            chunks.append(self.systems[head].forward(chunk, time))
        return concatenate(chunks, axis=-1)

    def states(self) -> List[MemoryState]:
        states: List[MemoryState] = []
        for system in self.systems:
            states.extend(system.states())
        return states
