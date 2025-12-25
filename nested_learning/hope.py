"""HOPE (Self-Mod) architecture based on Nested Learning paper (torch version)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    memory: torch.Tensor


class HOPEModel(nn.Module):
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
        super().__init__()
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
        self.state = HopeState(time=0, memory=torch.zeros(1, hidden_dim))
        self._last_context: torch.Tensor | None = None
        self._last_logits: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, time: int, update_memory: bool = True) -> torch.Tensor:
        encoded = F.relu(self.encoder(x))
        memory_context = self.cms.forward(encoded, time, update=update_memory)
        modulated = self.self_mod(memory_context)
        normed = self.norm(modulated)
        logits = self.decoder(normed)
        if update_memory:
            self.state = HopeState(time=time, memory=memory_context.detach())
            self._last_context = memory_context
            logits.retain_grad()
            self._last_logits = logits
        return logits

    def self_update(self, x: torch.Tensor, grad: torch.Tensor):
        self.self_mod.self_update(x, grad)

    def self_update_from_logits(self):
        if self._last_context is None or self._last_logits is None or self._last_logits.grad is None:
            return
        grad_hidden = self._last_logits.grad @ self.decoder.weight.t()
        self.self_mod.self_update(self._last_context, grad_hidden)

    def reset(self):
        self.state = HopeState(time=0, memory=torch.zeros_like(self.state.memory))
