"""HOPE (Self-Mod) architecture based on Nested Learning paper (torch version)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import AttentionBlock, Linear, LayerNorm, NestedContextFlow, SelfModifyingStack
from .memory import (
    ContinuumMemorySystem,
    NestedContinuumMemorySystem,
    SequentialContinuumMemorySystem,
    HeadwiseContinuumMemorySystem,
    MemoryBlock,
)


@dataclass
class HopeState:
    time: int
    memory: List[Any] | None  # Supports nested list of states
    decoder_memory: Any | None = None


class HOPEModel(nn.Module):
    @staticmethod
    def preset_config(preset: str) -> dict[str, Any]:
        if preset == "adaptive":
            return {
                "memory_decay": 0.05,
                "replay_ratio": 0.1,
                "replay_steps": 1,
                "self_mod_depth": 3,
                "nested_depth": 2,
                "nested_hidden": 128,
            }
        if preset == "fast_adapt":
            return {
                "memory_decay": 0.015,
                "replay_ratio": 0.0,
                "replay_steps": 0,
                "self_mod_depth": 4,
                "nested_depth": 2,
                "nested_hidden": 128,
            }
        if preset == "high_retention":
            return {
                "memory_decay": 0.01,
                "replay_ratio": 0.0,
                "replay_steps": 0,
                "self_mod_depth": 3,
                "nested_depth": 2,
                "nested_hidden": 128,
            }
        if preset == "balanced":
            return {
                "memory_decay": 0.01,
                "replay_ratio": 0.0,
                "replay_steps": 0,
                "self_mod_depth": 3,
                "nested_depth": 2,
                "nested_hidden": 128,
            }
        raise ValueError(f"Unknown preset: {preset}")

    @staticmethod
    def auto_scale_config(dataset_size: int | None, task_count: int | None) -> dict[str, Any]:
        if not dataset_size or not task_count:
            return {}
        replay_buffer = int(min(20000, max(512, dataset_size * task_count)))
        replay_ratio = min(0.65, 0.2 + 0.1 * task_count)
        memory_decay = min(0.05, 0.005 + 0.003 * math.log2(task_count + 1))
        return {
            "replay_buffer": replay_buffer,
            "replay_ratio": replay_ratio,
            "memory_decay": memory_decay,
        }

    @staticmethod
    def auto_scale_cms(
        dataset_size: int | None,
        task_count: int | None,
        *,
        backbone: str = "titans",
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        if not dataset_size or not task_count:
            return {}

        min_chunk_size = 32
        if batch_size:
            min_chunk_size = max(min_chunk_size, batch_size)
        if backbone == "attention":
            return {
                "cms_chunk_size": min_chunk_size,
                "cms_memory_chunk_size": max(min_chunk_size, min_chunk_size * 2),
            }

        steps_per_task = max(1, math.ceil(dataset_size / task_count))
        if batch_size:
            steps_per_task = max(1, math.ceil(steps_per_task / batch_size))

        def snap_pow2(value: float, minimum: int = 1) -> int:
            return max(minimum, int(2 ** round(math.log2(max(value, 1)))))

        scale = 2 + task_count
        base_min = 128
        slowest_target = max(base_min, int(round(steps_per_task * scale)))
        if dataset_size >= 10000 or task_count >= 4:
            slowest_target = max(slowest_target, 512)
        if dataset_size >= 50000 or task_count >= 8:
            slowest_target = max(slowest_target, 2048)
        slowest_chunk = min(8192, snap_pow2(slowest_target, minimum=128))

        chunk_base = max(min_chunk_size, min(64, slowest_chunk // 16))
        cms_chunk_size = max(min_chunk_size, min(64, chunk_base))
        cms_memory_chunk_size = max(cms_chunk_size, min(64, cms_chunk_size * 2))

        return {
            "cms_chunk_size": cms_chunk_size,
            "cms_memory_chunk_size": cms_memory_chunk_size,
        }

    @classmethod
    def from_preset(
        cls,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        preset: str = "balanced",
        dataset_size: int | None = None,
        task_count: int | None = None,
        auto_scale: bool = True,
        **overrides: Any,
    ) -> "HOPEModel":
        backbone = overrides.get("backbone", "titans")
        config = cls.preset_config(preset)
        if auto_scale:
            config.update(cls.auto_scale_config(dataset_size, task_count))
            if preset == "adaptive":
                config.update(
                    cls.auto_scale_cms(
                        dataset_size,
                        task_count,
                        backbone=backbone,
                        batch_size=overrides.get("batch_size"),
                    )
                )
        config.update(overrides)
        cms_chunk_size = config.pop("cms_chunk_size", None)
        cms_memory_chunk_size = config.pop("cms_memory_chunk_size", None)
        config.pop("batch_size", None)
        if preset != "adaptive" and "frequencies" not in config and "hope_levels" not in config:
            raise ValueError("frequencies (or hope_levels) must be provided when using non-adaptive presets")
        model = cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            task_count=task_count,
            **config,
        )
        model.cms_chunk_size = cms_chunk_size
        model.cms_memory_chunk_size = cms_memory_chunk_size
        return model

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        task_count: int | None = None,
        frequencies: List[int] | None = None,
        cms_depth: int = 2,
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
                frequencies = [1, 8, 16]
            else:
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
                depth=cms_depth,
                decay=memory_decay,
                replay_ratio=replay_ratio,
                replay_steps=replay_steps,
                replay_buffer=replay_buffer,
            )
        elif cms_variant == "sequential":
            self.cms = SequentialContinuumMemorySystem(
                hidden_dim,
                frequencies=frequencies,
                depth=cms_depth,
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
                depth=cms_depth,
                decay=memory_decay,
                replay_ratio=replay_ratio,
                replay_steps=replay_steps,
                replay_buffer=replay_buffer,
            )
        else:
            self.cms = ContinuumMemorySystem(
                hidden_dim,
                frequencies=frequencies,
                depth=cms_depth,
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
        self.cms_chunk_size: int | None = None
        self.cms_memory_chunk_size: int | None = None
        self._ema_loss: float | None = None
        self._ema_acc: float | None = None
        self._best_acc: float | None = None
        self._cms_autoscale_cooldown = 0
        self._last_frequencies: list[int] | None = None
        self.attention = AttentionBlock(hidden_dim, heads=heads) if backbone == "attention" else None
        # Memory-Augmented Decoder to protect mapping
        self.task_count = task_count
        if task_count:
            self.decoder_memories = nn.ModuleList(
                [
                    MemoryBlock(
                        hidden_dim,
                        frequency=1,
                        depth=1,
                        replay_ratio=replay_ratio,
                        replay_buffer=replay_buffer,
                    )
                    for _ in range(task_count)
                ]
            )
            self.decoder_heads = nn.ModuleList([Linear(hidden_dim, output_dim) for _ in range(task_count)])
            self.decoder_mem = None
            self.decoder = None
        else:
            self.decoder_mem = MemoryBlock(
                hidden_dim,
                frequency=1,
                depth=1,
                replay_ratio=replay_ratio,
                replay_buffer=replay_buffer,
            )
            self.decoder = Linear(hidden_dim, output_dim)
            self.decoder_memories = None
            self.decoder_heads = None
        self.norm = LayerNorm(hidden_dim)
        self.final_norm = LayerNorm(hidden_dim) # Protect decoder from signal explosion
        self.state = HopeState(time=0, memory=None, decoder_memory=None)

        self.replay_ratio = replay_ratio
        self.replay_steps = replay_steps
        self.replay_buffer_limit = replay_buffer
        self.raw_replay_buffer: dict[int, deque[tuple[torch.Tensor, torch.Tensor]]] = {}

        self._last_context: torch.Tensor | None = None
        self._last_logits: torch.Tensor | None = None
        self._last_task_id: int | None = None

    def _select_task(self, task_id: int | None) -> int:
        if self.task_count is None:
            if task_id is not None:
                raise ValueError("task_id provided but task_count is not set")
            return 0
        return task_id or 0

    def remember(self, x: torch.Tensor, y: torch.Tensor, task_id: int | None = None):
        # Store raw experience as tuples (x, y) per task
        buffer_key = task_id if task_id is not None else -1
        if buffer_key not in self.raw_replay_buffer:
            self.raw_replay_buffer[buffer_key] = deque(maxlen=self.replay_buffer_limit)
        x_detached = x.detach().cpu()
        y_detached = y.detach().cpu()
        for i in range(x.size(0)):
            self.raw_replay_buffer[buffer_key].append((x_detached[i], y_detached[i]))

    def sample_replay(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if not self.raw_replay_buffer:
            return None
        total_samples = sum(len(buf) for buf in self.raw_replay_buffer.values())
        if total_samples < batch_size:
            return None
        import random
        buffers = list(self.raw_replay_buffer.items())
        per_task = max(1, batch_size // len(buffers))
        samples: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for task_key, buf in buffers:
            if not buf:
                continue
            take = min(per_task, len(buf))
            samples.extend((task_key, x, y) for x, y in random.sample(buf, take))
        remaining = batch_size - len(samples)
        if remaining > 0:
            flat_buffer = [(task_key, x, y) for task_key, buf in buffers for x, y in buf]
            samples.extend(random.sample(flat_buffer, remaining))
        task_ids, x_list, y_list = zip(*samples)
        # Move to device of current parameters
        device = next(self.parameters()).device
        x_batch = torch.stack(x_list).to(device)
        y_batch = torch.stack(y_list).to(device)
        task_batch = torch.tensor(task_ids, device=device, dtype=torch.long)
        return x_batch, y_batch, task_batch

    def forward(
        self,
        x: torch.Tensor,
        time: int,
        update_memory: bool = True,
        task_id: int | None = None,
        detach_encoder: bool = False,
    ) -> torch.Tensor:
        memory_context, new_cms_states, encoded = self.forward_features(
            x,
            time=time,
            update_memory=update_memory,
            detach_encoder=detach_encoder,
        )
        logits, new_dec_state, task_index = self.forward_decoder(
            memory_context,
            task_id=task_id,
            update_memory=update_memory,
        )
        if update_memory:
            if encoded.requires_grad:
                encoded.retain_grad()
            self.state = HopeState(
                time=time,
                memory=[s.detach() if isinstance(s, torch.Tensor) else s for s in new_cms_states],
                decoder_memory=new_dec_state.detach(),
            )
            self._last_context = encoded
            logits.retain_grad()
            self._last_logits = logits
            self._last_task_id = task_index
        return logits

    def forward_features(
        self,
        x: torch.Tensor,
        time: int,
        update_memory: bool = True,
        detach_encoder: bool = False,
        state: HopeState | None = None,
    ) -> tuple[torch.Tensor, List[Any], torch.Tensor]:
        encoded = F.relu(self.encoder(x))
        if detach_encoder:
            encoded = encoded.detach()
        if self.pre_norm is not None:
            encoded = self.pre_norm(encoded)
        if self.conv is not None:
            if encoded.dim() == 2:
                encoded = self.conv(encoded.unsqueeze(-1)).squeeze(-1)
            else:
                encoded = self.conv(encoded.transpose(1, 2)).transpose(1, 2)

        if self.backbone == "attention":
            modulated = self.attention(encoded)
        else:
            modulated = self.self_mod(encoded)
        normed = self.norm(modulated)
        if self.post_norm is not None:
            normed = self.post_norm(normed)

        def check_batch_dim(states, batch_size):
            if states is None:
                return False
            if isinstance(states, (list, tuple)):
                if not states:
                    return True
                return check_batch_dim(states[0], batch_size)
            if isinstance(states, torch.Tensor):
                return states.size(0) == batch_size
            return True

        current_states = self.state.memory if state is None else state.memory

        if update_memory and self.cms_chunk_size:
            chunk_size = max(1, self.cms_chunk_size)
            memory_chunks = []
            new_cms_states = None
            for start in range(0, normed.size(0), chunk_size):
                chunk = normed[start : start + chunk_size]
                if not check_batch_dim(current_states, chunk.size(0)):
                    current_states = None
                memory_context, new_cms_states = self.cms.forward(
                    chunk,
                    time,
                    states=current_states,
                    update=True,
                )
                if self.nested_flow is not None:
                    memory_context = self.nested_flow.forward(memory_context, time, update=True)
                memory_chunks.append(memory_context)
                current_states = new_cms_states
            memory_context = torch.cat(memory_chunks, dim=0)
        else:
            if not check_batch_dim(current_states, normed.size(0)):
                current_states = None
            memory_context, new_cms_states = self.cms.forward(normed, time, states=current_states, update=update_memory)
            if self.nested_flow is not None:
                memory_context = self.nested_flow.forward(memory_context, time, update=update_memory)

        return memory_context, new_cms_states, encoded

    def forward_decoder(
        self,
        memory_context: torch.Tensor,
        task_id: int | None = None,
        update_memory: bool = True,
        state: HopeState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        task_index = self._select_task(task_id)

        if self.task_count:
            decoder_mem = self.decoder_memories[task_index]
            decoder = self.decoder_heads[task_index]
        else:
            decoder_mem = self.decoder_mem
            decoder = self.decoder

        dec_state = self.state.decoder_memory if state is None else state.decoder_memory
        if dec_state is not None and dec_state.size(0) != memory_context.size(0):
            dec_state = None
        if dec_state is None:
            dec_state = torch.zeros(
                memory_context.size(0),
                memory_context.size(1),
                device=memory_context.device,
                dtype=memory_context.dtype,
            )

        normed_mem, new_dec_state = decoder_mem.update(memory_context, dec_state, update=update_memory)
        logits = decoder(self.final_norm(normed_mem))
        return logits, new_dec_state, task_index

    def update_chunk(
        self,
        x: torch.Tensor,
        chunk_size: int | None = None,
        memory_chunk_size: int | None = None,
        task_id: int | None = None,
    ):
        def expand_state(state, batch_size: int):
            if state is None:
                return None
            if isinstance(state, torch.Tensor):
                if state.size(0) == batch_size:
                    return state
                state_mean = state.mean(dim=0, keepdim=True)
                return state_mean.expand(batch_size, -1).contiguous()
            if isinstance(state, (list, tuple)):
                expanded = [expand_state(item, batch_size) for item in state]
                return type(state)(expanded)
            return state

        # Flatten if 3D (Batch, Seq, Dim) -> (Batch*Seq, Dim)
        if x.dim() == 3:
            x = x.view(-1, x.shape[-1])

        # Replay mixing is handled externally now.
        # update_chunk simply processes the provided input x (which is already mixed).

        with torch.no_grad():
            encoded = F.relu(self.encoder(x))
            if self.pre_norm is not None:
                encoded = self.pre_norm(encoded)
            if self.conv is not None:
                if encoded.dim() == 2:
                    encoded = self.conv(encoded.unsqueeze(-1)).squeeze(-1)
                else:
                    encoded = self.conv(encoded.transpose(1, 2)).transpose(1, 2)

            if self.backbone == "attention":
                modulated = self.attention(encoded)
            else:
                modulated = self.self_mod.update_chunk_and_forward(
                    encoded,
                    chunk_size=chunk_size,
                    memory_chunk_size=memory_chunk_size,
                    projection_mask=self.self_mod_projection_mask,
                )

            normed = self.norm(modulated)
            if self.post_norm is not None:
                normed = self.post_norm(normed)

        # Step E: Update CMS
        # We pass update=True to allow CMS to learn from this mixed batch.
        batch_size = normed.size(0)
        expanded_states = expand_state(self.state.memory, batch_size)
        with torch.no_grad():
            memory_context, _ = self.cms.forward(normed, time=self.state.time, states=expanded_states, update=True)

        if self.nested_flow is not None:
            with torch.no_grad():
                memory_context = memory_context.detach()
                memory_context = self.nested_flow.forward(memory_context, time=self.state.time, update=True)

        task_index = self._select_task(task_id)
        if self.task_count:
            decoder_mem = self.decoder_memories[task_index]
        else:
            decoder_mem = self.decoder_mem

        # Reset state for chunk update, but preserve mean state if available.
        dec_state = expand_state(self.state.decoder_memory, memory_context.size(0))
        if dec_state is None:
            dec_state = torch.zeros(
                memory_context.size(0),
                memory_context.size(1),
                device=memory_context.device,
                dtype=memory_context.dtype,
            )
        decoder_mem.update(memory_context, dec_state, update=True)

    def _iter_cms_blocks(self):
        if hasattr(self.cms, "blocks"):
            for block in self.cms.blocks:
                yield block
        elif hasattr(self.cms, "systems"):
            for system in self.cms.systems:
                for block in system.blocks:
                    yield block

    def maybe_rescale_cms(
        self,
        *,
        loss: float,
        accuracy: float | None,
        dataset_size: int | None,
        task_count: int | None,
        batch_size: int | None = None,
        spike_threshold: float = 0.25,
        acc_drop: float = 0.05,
        cooldown_steps: int = 50,
    ) -> bool:
        if not dataset_size or not task_count:
            return False

        loss_value = float(loss)
        if self._ema_loss is None:
            self._ema_loss = loss_value
        else:
            self._ema_loss = 0.9 * self._ema_loss + 0.1 * loss_value

        if accuracy is not None:
            if self._ema_acc is None:
                self._ema_acc = accuracy
            else:
                self._ema_acc = 0.9 * self._ema_acc + 0.1 * accuracy
            if self._best_acc is None or accuracy > self._best_acc:
                self._best_acc = accuracy

        if self._cms_autoscale_cooldown > 0:
            self._cms_autoscale_cooldown -= 1
            return False

        loss_spike = self._ema_loss is not None and loss_value > self._ema_loss * (1.0 + spike_threshold)
        acc_drop_trigger = False
        if accuracy is not None and self._best_acc is not None:
            acc_drop_trigger = accuracy < (self._best_acc - acc_drop)

        if not (loss_spike or acc_drop_trigger):
            return False

        config = self.auto_scale_cms(
            dataset_size,
            task_count,
            backbone=self.backbone,
            batch_size=batch_size,
        )
        self.cms_chunk_size = config.get("cms_chunk_size", self.cms_chunk_size)
        self.cms_memory_chunk_size = config.get("cms_memory_chunk_size", self.cms_memory_chunk_size)
        self._cms_autoscale_cooldown = max(1, cooldown_steps)
        return True

    def self_update_from_logits(self):
        if self._last_context is None or self._last_logits is None or self._last_logits.grad is None:
            return
        if self.self_mod is None:
            return
        grad_hidden = None
        if isinstance(self._last_context, torch.Tensor) and self._last_context.grad is not None:
            grad_hidden = self._last_context.grad
        if self.task_count:
            if self._last_task_id is None:
                return
            decoder = self.decoder_heads[self._last_task_id]
        else:
            decoder = self.decoder
        if grad_hidden is None:
            grad_hidden = self._last_logits.grad @ decoder.weight.t()
        self.self_mod.update_chunk(self._last_context + grad_hidden)

    def reset(self):
        self.state = HopeState(time=0, memory=None, decoder_memory=None)
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
        if hasattr(self, "decoder_mem"):
             self.decoder_mem.step_counter = 0
