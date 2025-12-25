"""HOPE (Self-Mod) architecture based on Nested Learning paper (torch version)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

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
        self_mod_type: Literal["titan", "attention"] = "titan",
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
        self.self_mod_type = self_mod_type
        if self_mod_type == "attention":
            self.self_mod = nn.MultiheadAttention(hidden_dim, num_heads=heads, batch_first=True)
        else:
            self.self_mod = SelfModifyingStack(hidden_dim, depth=self_mod_depth)
        self.decoder = Linear(hidden_dim, output_dim)
        self.norm = LayerNorm(hidden_dim)
        self.state = HopeState(time=0, memory=torch.zeros(1, hidden_dim))
        self._last_context: torch.Tensor | None = None
        self._last_logits: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, time: int, update_memory: bool = True) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        encoded = F.relu(self.encoder(x))
        if self.self_mod_type == "attention":
            modulated, _ = self.self_mod(encoded, encoded, encoded, need_weights=False)
        else:
            modulated = self.self_mod(encoded)
        memory_context = self._cms_forward_sequence(modulated, time, update=update_memory)
        normed = self.norm(memory_context)
        logits = self.decoder(normed)
        if update_memory:
            self.state = HopeState(time=time, memory=memory_context.detach())
            self._last_context = memory_context
            logits.retain_grad()
            self._last_logits = logits
        if logits.shape[1] == 1:
            return logits.squeeze(1)
        return logits

    def update_chunk(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        encoded = F.relu(self.encoder(x))
        if self.self_mod_type == "attention":
            return
        self.self_mod.update_chunk(encoded.reshape(-1, encoded.shape[-1]))

    def self_update_from_logits(self):
        if self._last_context is None or self._last_logits is None or self._last_logits.grad is None:
            return
        grad_hidden = self._last_logits.grad @ self.decoder.weight.t()
        if self.self_mod_type == "attention":
            return
        self.self_mod.update_chunk((self._last_context + grad_hidden).reshape(-1, grad_hidden.shape[-1]))

    def reset(self):
        self.state = HopeState(time=0, memory=torch.zeros_like(self.state.memory))
        self.cms.reset()

    def _cms_forward_sequence(self, x: torch.Tensor, time: int, update: bool) -> torch.Tensor:
        context = []
        for offset in range(x.shape[1]):
            token = x[:, offset, :]
            ctx = self.cms.forward(token, time + offset, update=update)
            context.append(ctx)
        return torch.stack(context, dim=1)
