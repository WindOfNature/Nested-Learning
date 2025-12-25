"""HOPE (Self-Mod) architecture based on Nested Learning paper (torch version)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import AttentionBlock, Linear, LayerNorm, NestedContextFlow, SelfModifyingStack
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
        frequencies: List[int] | None = None,
        cms_variant: str = "nested",
        self_mod_depth: int = 2,
        heads: int = 4,
        self_mod_query_static: bool = False,
        self_mod_projection_mask: tuple[bool, bool, bool, bool, bool] | None = None,
        backbone: str = "titans",
        hope_levels: int | None = None,
        lowest_frequency: int = 1,
        nested_depth: int = 0,
        nested_hidden: int = 128,
        memory_decay: float = 0.0,
        replay_ratio: float = 0.0,
        replay_steps: int = 1,
        replay_buffer: int = 128,
        use_conv: bool = True,
        conv_kernel: int = 3,
        use_pre_norm: bool = True,
        use_post_norm: bool = True,
    ):
        super().__init__()
        if frequencies is None:
            if hope_levels is None:
                raise ValueError("frequencies or hope_levels must be provided")
            frequencies = [lowest_frequency * (2**idx) for idx in range(hope_levels)]
        self.encoder = Linear(input_dim, hidden_dim)
        self.pre_norm = LayerNorm(hidden_dim) if use_pre_norm else None
        self.post_norm = LayerNorm(hidden_dim) if use_post_norm else None
        self.use_conv = use_conv
        self.conv = (
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=conv_kernel, padding=conv_kernel // 2)
            if use_conv
            else None
        )
        if cms_variant == "nested":
            self.cms = NestedContinuumMemorySystem(
                hidden_dim,
                frequencies=frequencies,
                decay=memory_decay,
                replay_ratio=replay_ratio,
                replay_steps=replay_steps,
                replay_buffer=replay_buffer,
            )
        elif cms_variant == "sequential":
            self.cms = SequentialContinuumMemorySystem(
                hidden_dim,
                frequencies=frequencies,
                decay=memory_decay,
                replay_ratio=replay_ratio,
                replay_steps=replay_steps,
                replay_buffer=replay_buffer,
            )
        elif cms_variant == "headwise":
            self.cms = HeadwiseContinuumMemorySystem(
                hidden_dim,
                frequencies=frequencies,
                heads=heads,
                decay=memory_decay,
                replay_ratio=replay_ratio,
                replay_steps=replay_steps,
                replay_buffer=replay_buffer,
            )
        else:
            self.cms = ContinuumMemorySystem(
                hidden_dim,
                frequencies=frequencies,
                decay=memory_decay,
                replay_ratio=replay_ratio,
                replay_steps=replay_steps,
                replay_buffer=replay_buffer,
            )
        self.nested_flow = NestedContextFlow(hidden_dim, depth=nested_depth, hidden=nested_hidden) if nested_depth else None
        self.backbone = backbone
        self.self_mod_projection_mask = self_mod_projection_mask
        self.self_mod = (
            SelfModifyingStack(hidden_dim, depth=self_mod_depth, query_static=self_mod_query_static)
            if backbone == "titans"
            else None
        )
        self.attention = AttentionBlock(hidden_dim, heads=heads) if backbone == "attention" else None
        self.decoder = Linear(hidden_dim, output_dim)
        self.norm = LayerNorm(hidden_dim)
        self.state = HopeState(time=0, memory=torch.zeros(1, hidden_dim))
        self._last_context: torch.Tensor | None = None
        self._last_logits: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, time: int, update_memory: bool = True) -> torch.Tensor:
        encoded = F.relu(self.encoder(x))
        if self.pre_norm is not None:
            encoded = self.pre_norm(encoded)
        if self.conv is not None:
            if encoded.dim() == 2:
                encoded = self.conv(encoded.unsqueeze(-1)).squeeze(-1)
            else:
                encoded = self.conv(encoded.transpose(1, 2)).transpose(1, 2)
        memory_context = self.cms.forward(encoded, time, update=update_memory)
        if self.nested_flow is not None:
            memory_context = self.nested_flow.forward(memory_context, time, update=update_memory)
        if self.backbone == "attention":
            modulated = self.attention(memory_context)
        else:
            modulated = self.self_mod(memory_context)
        normed = self.norm(modulated)
        if self.post_norm is not None:
            normed = self.post_norm(normed)
        logits = self.decoder(normed)
        if update_memory:
            self.state = HopeState(time=time, memory=memory_context.detach())
            self._last_context = memory_context
            logits.retain_grad()
            self._last_logits = logits
        return logits

    def update_chunk(
        self,
        x: torch.Tensor,
        chunk_size: int | None = None,
        memory_chunk_size: int | None = None,
    ):
        encoded = F.relu(self.encoder(x))
        if self.pre_norm is not None:
            encoded = self.pre_norm(encoded)
        if self.conv is not None:
            if encoded.dim() == 2:
                encoded = self.conv(encoded.unsqueeze(-1)).squeeze(-1)
            else:
                encoded = self.conv(encoded.transpose(1, 2)).transpose(1, 2)
        if self.nested_flow is not None:
            encoded = self.nested_flow.forward(encoded, time=self.state.time, update=True)
        if self.self_mod is not None:
            self.self_mod.update_chunk(
                encoded,
                chunk_size=chunk_size,
                memory_chunk_size=memory_chunk_size,
                projection_mask=self.self_mod_projection_mask,
            )

    def self_update_from_logits(self):
        if self._last_context is None or self._last_logits is None or self._last_logits.grad is None:
            return
        grad_hidden = self._last_logits.grad @ self.decoder.weight.t()
        self.self_mod.update_chunk(self._last_context + grad_hidden)

    def reset(self):
        self.state = HopeState(time=0, memory=torch.zeros_like(self.state.memory))
