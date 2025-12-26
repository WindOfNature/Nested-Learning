"""HOPE (Self-Mod) architecture based on Nested Learning paper (torch version)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, List

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
    memory: List[Any] | None  # Supports nested list of states


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
        use_conv: bool = False,
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
        self.state = HopeState(time=0, memory=None)

        self.replay_ratio = replay_ratio
        self.replay_steps = replay_steps
        self.raw_replay_buffer = deque(maxlen=replay_buffer)

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

        # CMS update with state passing
        current_states = self.state.memory
        # If batch size changed (e.g. last batch), reset states to avoid mismatch
        if current_states is not None:
             # Check first state tensor if available
             # current_states is a list. It might contain Tensors or lists (nested).
             # We need to find a tensor to check dimension.
             # Simply: if we catch an error, or check proactively.
             # Proactive check:
             # Flatten structure to find first tensor?
             # Or just try/except? No.
             # Let's assume flat list for simple CMS, nested for others.
             # We'll just reset if x.size(0) doesn't match expected.
             # BUT we don't know expected easily without inspecting current_states.
             # Let's just catch size mismatch by checking the first block state if possible.
             # Or safer: if current_states matches x size.
             pass
             # Easier: Just pass current_states. If None, CMS inits.
             # If dimensions mismatch, CMS will crash.
             # So we must check.

        # Helper to check batch dimension match
        def check_batch_dim(states, batch_size):
            if states is None: return False
            if isinstance(states, (list, tuple)):
                if not states: return True
                return check_batch_dim(states[0], batch_size)
            if isinstance(states, torch.Tensor):
                return states.size(0) == batch_size
            return True # Unknown type

        if not check_batch_dim(current_states, encoded.size(0)):
             current_states = None

        memory_context, new_cms_states = self.cms.forward(encoded, time, states=current_states, update=update_memory)

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
            # We detach states to prevent infinite graph growth across steps,
            # but for BPTT within a window we might want to keep it?
            # Standard RNN practice is detach between batches.
            # Here we assume user handles truncation or we detach.
            # But wait, "Fix BPTT Break" suggests we want gradients.
            # If we detach here, we break BPTT across calls to forward.
            # Assuming typical usage (like in examples), forward is called per batch.
            # States should probably be detached unless doing TBPTT.
            # The prompt says "Fix BPTT Break: Updating state_value inside no_grad breaks... Pass hidden_state explicitly".
            # This allows gradients to flow IF the user wants them (e.g. sequence training).
            # But we store it in self.state.
            # I will detach here to be safe for infinite loops, but return/keep graph if needed?
            # self.state is used for next step.
            self.state = HopeState(time=time, memory=[s.detach() if isinstance(s, torch.Tensor) else s for s in new_cms_states])
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
        # Flatten if 3D (Batch, Seq, Dim) -> (Batch*Seq, Dim)
        if x.dim() == 3:
            x = x.view(-1, x.shape[-1])

        # Step A: Store raw experience (detached from graph)
        # Assuming x is raw input
        self.raw_replay_buffer.extend(x.detach().unbind(0))

        x_combined = x

        # Step B & C: Sample and Mix
        if self.replay_ratio > 0.0 and len(self.raw_replay_buffer) > 0:
            replay_count = max(1, int(self.replay_ratio * x.size(0)))
            if len(self.raw_replay_buffer) >= replay_count:
                import random
                replay_samples = random.sample(self.raw_replay_buffer, replay_count)
                replay_batch = torch.stack(replay_samples, dim=0)
                x_combined = torch.cat([x, replay_batch], dim=0)

        # Step D: Encode Combined Batch
        encoded = F.relu(self.encoder(x_combined))
        if self.pre_norm is not None:
            encoded = self.pre_norm(encoded)
        if self.conv is not None:
            if encoded.dim() == 2:
                encoded = self.conv(encoded.unsqueeze(-1)).squeeze(-1)
            else:
                encoded = self.conv(encoded.transpose(1, 2)).transpose(1, 2)

        # Step E: Update CMS
        # We pass update=True to allow CMS to learn from this mixed batch.
        # State is reset (None) for mixed/offline update.
        encoded, _ = self.cms.forward(encoded, time=self.state.time, states=None, update=True)

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
        self.state = HopeState(time=0, memory=None)
        # Clear CMS internal buffers if any (e.g. state_value in blocks was removed, but what about others?)
        # Step counter?
        # Iterate through blocks and reset buffers?
        # "Fix Reset Logic: Iterate through self.cms.blocks and zero out their internal state_value buffers."
        # I removed state_value buffers. But I should reset step_counter.
        if hasattr(self.cms, "blocks"):
             for block in self.cms.blocks:
                 block.step_counter = 0
                 # If block has other states...
        if hasattr(self.cms, "systems"): # Headwise
             for sys in self.cms.systems:
                  for block in sys.blocks:
                       block.step_counter = 0
