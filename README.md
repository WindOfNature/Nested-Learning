# Nested Learning

A production-grade implementation of **Nested Learning (NL)** and **HOPE** from
*"Nested Learning: The Illusion of Deep Learning Architecture"* (Behrouz et al.).
This package provides a torch-native framework for nested optimization, self-modifying
associative memory (Titans), a Continuum Memory System (CMS), and optimized CPU/GPU
kernels.

---

## Highlights

- **Nested context-flow levels** for explicit multi-level optimization.
- **Self-referential Titans** with adaptive k/v/q/η/α memories and self-modifying updates.
- **Continuum Memory System (CMS)** with configurable frequency, consolidation (decay), and replay.
- **Expressive optimizers** with NL-style associative-memory steering on top of `torch.optim`.
- **CPU (Numba) and GPU (Triton) kernels** for matmul and layernorm.

---

## Installation

```bash
pip install -e .
```

Optional training extras:

```bash
pip install -e .[train]
```

---

## Quickstarts

### 1) Basic HOPE forward pass

```python
import torch
from nested_learning.hope import HOPEModel

model = HOPEModel(
    input_dim=64,
    hidden_dim=128,
    output_dim=10,
    frequencies=[1, 4, 16],
    cms_variant="nested",
    self_mod_depth=3,
    nested_depth=2,
    memory_decay=0.1,
    replay_ratio=0.2,
)

x = torch.randn(8, 64)
logits = model(x, time=0)
```

### 2) Continual learning demo (digits)

```bash
python examples/continual_digits.py \
  --epochs 2 \
  --batch-size 32 \
  --max-samples 500 \
  --chunk-size 4 \
  --memory-chunk-size 8 \
  --steered-optim \
  --precondition outer
```

### 3) Steered optimizer (wrapping torch.optim)

```python
import torch
from nested_learning.torch_optim import ContextSteeredOptimizer, SteeredOptimizerConfig

model = torch.nn.Linear(16, 4)
config = SteeredOptimizerConfig(
    precondition="outer",  # none | adagrad | adam | outer
    memory_beta=0.9,
    variance_beta=0.999,
    alpha=0.5,
    weight_decay=1e-3,
)
optimizer = ContextSteeredOptimizer(model.parameters(), torch.optim.AdamW, config=config, lr=1e-3)
```

---

## Architecture Overview

### Self-Referential Titans
Titans generate **k, v, q, η, α** from input context and **self-modify** via memory modules:

- `M_k`, `M_v`, `M_q`, `M_η`, `M_α`, `M_memory` are 2-layer MLP memories.
- Each memory is trained via **DGD** with chunk-wise updates.
- Optional **static query** projection (`self_mod_query_static=True`) mirrors the paper's
  non-adaptive `W_q` path.

### Nested Context-Flow
`NestedContextFlow` provides explicit **level-wise context optimization**:

- Each level updates its own transformation using DGD.
- Levels can be stacked in `HOPEModel` (`nested_depth > 0`).

### Continuum Memory System (CMS)
CMS is a chain (or nested/sequence/headwise variant) of memory blocks with distinct
update frequencies and consolidation controls:

- `decay`: exponential consolidation of memory states.
- `replay_ratio`: ratio of replayed samples per update.
- `replay_steps`: repeat replay for stronger offline consolidation.

---

## API Reference

### `HOPEModel`

```python
HOPEModel(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    frequencies: list[int],
    cms_variant: str = "nested",  # nested | sequential | headwise | chain
    self_mod_depth: int = 2,
    heads: int = 4,
    self_mod_query_static: bool = False,
    nested_depth: int = 0,
    nested_hidden: int = 128,
    memory_decay: float = 0.0,
    replay_ratio: float = 0.0,
    replay_steps: int = 1,
    replay_buffer: int = 128,
)
```

- `forward(x, time, update_memory=True)`
- `update_chunk(x, chunk_size=None, memory_chunk_size=None)`
- `self_update_from_logits()` for self-modification via logits gradients

---

### `ContextSteeredOptimizer`

Wraps any base optimizer with associative-memory steering.

```python
ContextSteeredOptimizer(
    params,
    base_optimizer,
    config: SteeredOptimizerConfig,
    **base_optimizer_kwargs
)
```

`SteeredOptimizerConfig`:

- `precondition`: `none`, `adagrad`, `adam`, `outer`
- `memory_beta`, `variance_beta`, `alpha`, `eps`, `weight_decay`

---

## Kernels

- **CPU**: Numba-accelerated matmul + layernorm (`nested_learning.kernels.cpu`)
- **GPU**: Triton matmul + layernorm (`nested_learning.kernels.gpu`)

LayerNorm will dispatch to Triton when CUDA is available and gradients are disabled.

---

## Package Layout

```
nested_learning/
  hope.py            # HOPE architecture
  nn.py              # Titans, ContextFlow, core layers
  memory.py          # CMS variants
  torch_optim.py     # DGD + steered optimizer wrapper
  kernels/
    cpu.py           # Numba CPU kernels
    gpu.py           # Triton GPU kernels
examples/
  continual_digits.py
```

---

## Notes

- This implementation is designed to mirror the Nested Learning paper and
  provide a flexible, extensible framework for experimentation.
- For best performance, install CUDA + Triton for GPU kernels.

---

## License

Apache-2.0
