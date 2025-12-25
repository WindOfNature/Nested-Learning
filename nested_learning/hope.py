"""HOPE (Self-Mod) architecture based on Nested Learning paper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .tensor import Tensor
from .nn import Linear, LayerNorm, SelfModifyingStack
from .memory import (
    ContinuumMemorySystem,
    NestedContinuumMemorySystem,
    SequentialContinuumMemorySystem,
    HeadwiseContinuumMemorySystem,
)


@dataclass
class HopeState:
    time: int
    memory: Tensor


class HOPEModel:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        frequencies: List[int],
        cms_variant: str = "nested",
        self_mod_depth: int = 2,
        heads: int = 4,
    ):
        self.encoder = Linear(input_dim, hidden_dim)
        if cms_variant == "nested":
            self.cms = NestedContinuumMemorySystem(hidden_dim, frequencies=frequencies)
        elif cms_variant == "sequential":
            self.cms = SequentialContinuumMemorySystem(hidden_dim, frequencies=frequencies)
        elif cms_variant == "headwise":
            self.cms = HeadwiseContinuumMemorySystem(hidden_dim, frequencies=frequencies, heads=heads)
        else:
            self.cms = ContinuumMemorySystem(hidden_dim, frequencies=frequencies)
        self.self_mod = SelfModifyingStack(hidden_dim, depth=self_mod_depth)
        self.decoder = Linear(hidden_dim, output_dim)
        self.norm = LayerNorm(hidden_dim)
        self.state = HopeState(time=0, memory=Tensor.zeros((1, hidden_dim)))
        self._last_context: Tensor | None = None
        self._last_logits: Tensor | None = None

    def parameters(self):
        params = []
        for module in [self.encoder, self.self_mod, self.decoder, self.norm]:
            params.extend(module.parameters())
        cms_blocks = []
        if hasattr(self.cms, "blocks"):
            cms_blocks.extend(self.cms.blocks)
        if hasattr(self.cms, "subsystems"):
            for subsystem in self.cms.subsystems:
                cms_blocks.extend(subsystem.blocks)
        if hasattr(self.cms, "systems"):
            for system in self.cms.systems:
                cms_blocks.extend(system.blocks)
        for block in cms_blocks:
            for layer in block.layers:
                params.extend(layer.parameters())
            params.extend(block.norm.parameters())
        return params

    def forward(self, x: Tensor, time: int) -> Tensor:
        encoded = self.encoder(x).relu()
        memory_context = self.cms.forward(encoded, time)
        modulated = self.self_mod(memory_context)
        normed = self.norm(modulated)
        logits = self.decoder(normed)
        self.state = HopeState(time=time, memory=memory_context.detach())
        self._last_context = memory_context
        self._last_logits = logits
        return logits

    def self_update(self, x: Tensor, grad: Tensor):
        self.self_mod.self_update(x, grad)

    def self_update_from_logits(self):
        if self._last_context is None or self._last_logits is None or self._last_logits.grad is None:
            return
        grad_hidden = self._last_logits.grad @ self.decoder.weight.data.T
        self.self_mod.self_update(self._last_context, Tensor(grad_hidden))

    def reset(self):
        self.state = HopeState(time=0, memory=Tensor.zeros(self.state.memory.data.shape))
