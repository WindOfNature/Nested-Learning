# Nested Learning

A production-grade implementation of **Nested Learning (NL)** and **HOPE** from
*"Nested Learning: The Illusion of Deep Learning Architecture"* (Behrouz et al.).
This repository exposes self-referential Titans, nested context-flow, a Continuum
Memory System (CMS), and expressive optimizers that follow the NL framework.

---

## Why Nested Learning
Nested Learning frames training as **multi-level, self-referential optimization**:
models learn to compress context into memory, and optimizers themselves become
associative memory modules. This codebase provides the machinery to build models
with explicit multi-timescale updates, self-modifying behavior, and structured
memory consolidation.

---

## Key Features

- **Self-Referential Titans** with adaptive **k, v, q, η, α, memory** modules.
- **Nested Context-Flow** (multi-level optimization) with per-level dynamics.
- **Continuum Memory System (CMS)** with configurable update frequencies,
  consolidation decay, and replay buffers.
- **Associative Memory** with **parametric** and **non-parametric** (attention-like)
  modes, including **Hebbian / Delta / Oja** learning rules.
- **Expressive Optimizers**: DGD, GGD, GM, and a steered wrapper over `torch.optim`.
- **CPU (Numba) + GPU (Triton)** kernels for matmul and layernorm.

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
PYTHONPATH=. python examples/continual_digits.py \
  --epochs 2 \
  --batch-size 32 \
  --max-samples 500 \
  --chunk-size 4 \
  --memory-chunk-size 8 \
  --steered-optim \
  --precondition outer
```

### 3) Steered optimizer (wrap any torch optimizer)

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

### 4) Associative memory (parametric or non-parametric)

```python
import torch
from nested_learning.nn import AssociativeMemory

keys = torch.randn(8, 32)
values = torch.randn(8, 32)
queries = torch.randn(4, 32)

mem = AssociativeMemory(32, mode="nonparametric")
outputs = mem(keys, values, queries)
```

---

## Architecture Overview

### Self-Referential Titans
Titans generate **k, v, q, η, α** from context and self-modify via memory modules.
Each memory module follows the 2-layer structure described in the paper
(Eqs. 89–91):

```
M□(x) = x + W□,1 σ(W□,2 x)
```

Chunk-wise updates follow the NL dual-form update scheduling (Sec. 8.2):
- All signals for a chunk are computed *before* updates.
- Projection memories and memory modules can use distinct chunk sizes.

### Nested Context-Flow
`NestedContextFlow` adds explicit **multi-level context optimization**:

- Each level performs its own update step and normalizes the resulting context.
- Useful for composing HOPE with deeper nested learning dynamics.

### Continuum Memory System (CMS)
CMS is a chain (or nested/sequence/headwise variant) of memory blocks with distinct
update frequencies and consolidation controls:

- `decay`: exponential consolidation (online → stable memory).
- `replay_ratio`: offline replay sampling.
- `replay_steps`: additional replay passes.

### Associative Memory & Learning Rules
The library implements associative memory as in the paper’s formulation:

- **Non-parametric** (Softmax/Nadaraya-Watson, Eq. 62–63).
- **Parametric** (Titans-style memory modules).
- Learning rules: **Hebbian**, **Delta**, **Oja** (Eq. 64–67).

---

## Optimizers

### DGD (Delta Gradient Descent)
Implements the paper’s DGD learning rule for memory modules.

### GGD (Generalized Gradient Descent)
Implements the self-referential update formulation (Eq. 59–60), where values
are generated from the model state itself.

### GM (Generalized Momentum)
Extends GGD with explicit momentum memory, preserving knowledge across phases.

### ContextSteeredOptimizer
Wraps any `torch.optim` optimizer and steers gradients via associative-memory
preconditioning:

- `none`, `adagrad`, `adam`, `outer` (outer-product preconditioning).

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
- `self_update_from_logits()`

---

## Optimizer State Persistence (Continual Learning)

You can preserve optimizer memory across tasks:

```python
from nested_learning.torch_optim import save_optimizer_state, load_optimizer_state

save_optimizer_state(optimizer, "optim_state.pt")
load_optimizer_state(optimizer, "optim_state.pt")
```

---

## Kernels

- **CPU**: Numba-accelerated matmul + layernorm (`nested_learning.kernels.cpu`)
- **GPU**: Triton matmul + layernorm (`nested_learning.kernels.gpu`)

LayerNorm dispatches to Triton when CUDA is available and gradients are disabled.

---

## Package Layout

```
nested_learning/
  hope.py            # HOPE architecture
  nn.py              # Titans, AssociativeMemory, ContextFlow, core layers
  memory.py          # CMS variants
  torch_optim.py     # DGD / GGD / GM + steered optimizer
  kernels/
    cpu.py           # Numba CPU kernels
    gpu.py           # Triton GPU kernels
examples/
  continual_digits.py
```

---

## License

Apache-2.0
