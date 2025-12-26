"""Continuum Memory System (CMS) implementation using torch."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, List, Sequence

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
        replay_ratio: float = 0.25,
        replay_steps: int = 1,
        replay_buffer: int = 128,
    ):
        super().__init__()
        self.frequency = frequency
        self.layers = nn.ModuleList([Linear(features, features) for _ in range(depth)])
        self.norm = LayerNorm(features)
        self.decay = decay
        self.replay_ratio = replay_ratio
        self.replay_steps = replay_steps
        self.replay_buffer: Deque[torch.Tensor] = deque(maxlen=replay_buffer)
        self.meta = MemoryMLP(features)
        self.malpha = MemoryMLP(features)
        # Only optimize memory content parameters (layers, norm) in the inner loop.
        # Meta-parameters (meta, malpha) are optimized by the outer loop (Task Loss).
        fast_params = list(self.layers.parameters()) + list(self.norm.parameters())
        self.optimizer = DGD(fast_params, lr=1e-3, beta=0.9, alpha=0.5, weight_decay=1e-4)
        self.step_counter = 0

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Memory content generation (MLP)
        mem = x
        for layer in self.layers:
            mem = F.relu(layer(mem))
        mem = self.norm(mem)

        # Meta-learning signals
        eta = torch.sigmoid(self.meta(x)) * 0.1
        alpha = torch.sigmoid(self.malpha(x))

        # Gated residual update
        out = x + eta * mem

        # Continuum state update (BPTT friendly)
        if self.decay > 0.0:
            new_hidden = (1.0 - self.decay) * hidden_state + self.decay * mem
        else:
            new_hidden = mem

        return out, new_hidden, eta, alpha, mem

    def update(self, x: torch.Tensor, hidden_state: torch.Tensor, update: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        if not update:
            # Just forward pass
            out, new_hidden, _, _, _ = self.forward(x, hidden_state)
            return out, new_hidden

        self.step_counter += 1

        # Forward pass with gradients
        out, new_hidden, eta_tensor, alpha_tensor, mem = self.forward(x, hidden_state)

        if self.step_counter % self.frequency == 0:
            # We need to perform optimization step.
            # DGD update requires gradients on parameters.
            # We compute loss on the output vs input (reconstruction/prediction)
            # The prompt implies cheap meta learning via eta in forward,
            # but we also need to update the weights of the block.

            # Note: We use the tensors from forward pass to preserve graph if needed,
            # but DGD.step takes scalar overrides usually.
            # However, prompt says "eta needs a direct gradient path" via out = x + eta * memory.
            # That path is established in forward().
            # Here we update the weights of the block using DGD.

            # Loss for weight update
            loss = F.mse_loss(out, x) # Auto-encoder style? Or prediction?

            if self.replay_ratio > 0.0 and self.replay_buffer:
                replay_count = max(1, int(self.replay_ratio * len(self.replay_buffer)))
                if len(self.replay_buffer) >= replay_count:
                    # Buffer stores samples (Tensor[Dim])
                    replay_samples = random.sample(self.replay_buffer, replay_count)
                    replay_batch = torch.stack(replay_samples, dim=0)

                    # Replay forward pass with zero state (assuming independence)
                    # Note: We need a new state tensor matching replay_batch size
                    replay_zeros = torch.zeros(replay_batch.size(0), self.norm.gamma.size(0), device=replay_batch.device)

                    # We only care about weight updates, so we ignore hidden state output
                    replay_out, _, _, _, _ = self.forward(replay_batch, replay_zeros)
                    loss = loss + F.mse_loss(replay_out, replay_batch)

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True) # Retain graph because out is used downstream?

            # Extract scalar values for optimizer (or mean)
            eta_val = eta_tensor.mean().item()
            alpha_val = alpha_tensor.mean().item()

            # Fix retention logic: decay = (1.0 - alpha) * scale
            weight_decay = (1.0 - alpha_val) * 1e-4

            self.optimizer.step(lr_override=eta_val, weight_decay_override=weight_decay)

        # Update replay buffer (detached)
        if self.replay_ratio > 0.0:
             # Store samples, not batches
             self.replay_buffer.extend(x.detach().unbind(0))

        # Return output with detached memory content to avoid in-place error during backprop
        # because 'layers' (used to compute mem) were updated in-place.
        # But we keep 'eta_tensor' attached so it can learn via Task Loss.
        out_detached = x + eta_tensor * mem.detach()

        # We also need to fix new_hidden if it depends on mem
        if self.decay > 0.0:
            new_hidden_detached = (1.0 - self.decay) * hidden_state + self.decay * mem.detach()
        else:
            new_hidden_detached = mem.detach()

        return out_detached, new_hidden_detached

    def state(self) -> MemoryState:
        # Placeholder as state is now external
        return MemoryState(time=0, value=torch.zeros(1))


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

    def forward(self, x: torch.Tensor, time: int, states: List[torch.Tensor] | None = None, update: bool = True) -> tuple[torch.Tensor, List[torch.Tensor]]:
        context = x
        new_states = []
        if states is None:
            states = [torch.zeros(x.size(0), self.features, device=x.device) for _ in self.blocks]

        for block, state in zip(self.blocks, states):
            # Chain: context = block.forward(context) which is handled by update returning out
            out, new_state = block.update(context, state, update=update)
            context = out # Compositional chain
            new_states.append(new_state)

        return context, new_states

    def states(self) -> List[MemoryState]:
        return [block.state() for block in self.blocks]


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

    def forward(self, x: torch.Tensor, time: int, states: List[Any] | None = None, update: bool = True) -> tuple[torch.Tensor, List[Any]]:
        context = x
        new_states = []
        if states is None:
            # Need recursive state init?
            # State structure: [(block_state, sub_system_states), ...]
            states = [None] * len(self.blocks)

        for idx, (block, sub) in enumerate(zip(self.blocks, self.subsystems)):
            current_state_pair = states[idx]
            if current_state_pair is None:
                block_state = torch.zeros(x.size(0), self.features, device=x.device)
                sub_states = None
            else:
                block_state, sub_states = current_state_pair

            out, new_block_state = block.update(context, block_state, update=update)

            # Nested context flow: sub system processes the output of the block?
            # Original: nested_context = sub.forward(context...) -> context + block.state + nested
            # New chain logic: context -> block -> sub -> context?
            # Or context -> (block + sub)?
            # "Fix CMS Chain ... Change to context = block.forward(context)" applies to linear chain.
            # Nested is tricky. "Fully nested CMS where each block owns a sub-CMS."
            # Maybe: context = sub(block(context))?

            nested_out, new_sub_states = sub.forward(out, time, states=sub_states, update=update)

            context = nested_out # Compositional
            new_states.append((new_block_state, new_sub_states))

        return context, new_states

    def states(self) -> List[MemoryState]:
        states: List[MemoryState] = []
        for block, sub in zip(self.blocks, self.subsystems):
            states.append(block.state())
            states.extend(sub.states())
        return states


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

    def forward(self, x: torch.Tensor, time: int, states: List[torch.Tensor] | None = None, update: bool = True) -> tuple[torch.Tensor, List[torch.Tensor]]:
        context = x
        new_states = []
        if states is None:
            states = [torch.zeros(x.size(0), self.features, device=x.device) for _ in self.blocks]

        for block, state in zip(self.blocks, states):
            out, new_state = block.update(context, state, update=update)
            # Sequential uses norm on the output
            context = self.norm(out)
            new_states.append(new_state)

        return context, new_states

    def states(self) -> List[MemoryState]:
        return [block.state() for block in self.blocks]


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

    def forward(self, x: torch.Tensor, time: int, states: List[Any] | None = None, update: bool = True) -> tuple[torch.Tensor, List[Any]]:
        chunks = []
        new_states = []
        if states is None:
            states = [None] * self.heads

        for head in range(self.heads):
            start = head * self.head_dim
            end = start + self.head_dim
            chunk = x[:, start:end]

            head_state = states[head]
            out, new_head_state = self.systems[head].forward(chunk, time, states=head_state, update=update)
            chunks.append(out)
            new_states.append(new_head_state)

        return torch.cat(chunks, dim=-1), new_states

    def states(self) -> List[MemoryState]:
        states: List[MemoryState] = []
        for system in self.systems:
            states.extend(system.states())
        return states
