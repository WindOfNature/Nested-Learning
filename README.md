# Nested Learning

An unofficial implementation of **Nested Learning (NL)** and **HOPE** from
*"Nested Learning: The Illusion of Deep Learning Architecture"* (Ali Behrouz et al, and other Google Researcher).
This repository provides self-referential Titans, nested context-flow, a Continuum
Memory System (CMS), expressive optimizers, and optimized CPU/GPU kernels.

---

## Disclaimer
This codebase was generated with assistance from OpenAI Codex (GPT-5.2-Codex-Max)
and may contain inaccuracies/imcomplete relative to the paper. Issues and PRs are greatly welcome to improve this framework further 😁.

---

## Highlights

- **Self-Referential Titans** with adaptive **k, v, q, η, α, memory** modules.
- **Hope-Attention** variant (Titans swapped for softmax attention).
- **Nested Context-Flow** for explicit multi-level optimization.
- **Continuum Memory System (CMS)** with multi-timescale update, decay, and replay.
- **Associative Memory** with parametric & non-parametric modes and **Hebbian / Delta / Oja** rules.
- **Expressive optimizers**: DGD, GGD, GM, plus a steered wrapper over `torch.optim`.
- **Note**: Custom kernels are currently disabled to ensure correct Autograd behavior during Meta-Learning.

---

## Installation

### 1) Local editable install

```bash
pip install -e .
```

### 2) From GitHub

```bash
pip install git+https://github.com/WindOfNature/Nested-Learning.git
```

### 3) Training extras

```bash
pip install -e .[train]
```

### 4) Optional GPU kernels (Triton)

Triton requires CUDA and a compatible PyTorch build. Once installed, the kernels
are automatically used during inference when gradients are disabled.

---

## Quickstarts

### A) HOPE (Titans backbone)

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
    backbone="titans",
)

x = torch.randn(8, 64)
logits = model(x, time=0)
```

### B) Hope-Attention

```python
import torch
from nested_learning.hope import HOPEModel

model = HOPEModel(
    input_dim=64,
    hidden_dim=128,
    output_dim=10,
    frequencies=[1, 4, 16],
    cms_variant="nested",
    backbone="attention",
)

x = torch.randn(8, 64)
logits = model(x, time=0)
```

### C) Continual learning demo (digits)

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

### D) CMS multi-level frequencies

```python
from nested_learning.hope import HOPEModel

model = HOPEModel(
    input_dim=64,
    hidden_dim=128,
    output_dim=10,
    hope_levels=3,
    lowest_frequency=2,
    cms_variant="nested",
)
```

### E) Projection-frequency ablation (freeze k/v/q)

```bash
PYTHONPATH=. python examples/continual_digits.py \
  --freeze-k --freeze-v --freeze-q
```

---

## Architecture Overview

### Self-Referential Titans
Titans generate **k, v, q, η, α** from context and self-modify via memory modules.
Each memory module follows the 2-layer structure from the paper (Eqs. 89–91):

```
M□(x) = x + W□,1 σ(W□,2 x)
```

Chunk-wise updates follow NL dual-form scheduling (Sec. 8.2):
- Signals for a chunk are computed *before* updates.
- Projection updates and memory updates can use different chunk sizes.

### Hope-Attention
Hope-Attention replaces the Titans block with a softmax attention module.
This matches the paper’s Hope-Attention ablation (Sec. 8.3).

### Continuum Memory System (CMS)
CMS is a chain (or nested/sequence/headwise variant) of memory blocks with
multi-timescale updates, consolidation, and replay:

- `decay`: exponential consolidation into stable memory.
- `replay_ratio`: offline replay sampling.
- `replay_steps`: extra replay passes.

### Normalization + Convolution
HOPE can include pre/post LayerNorms and a local convolution block, matching
Figure 5 / Section 8 in the paper. These are configurable via `use_pre_norm`,
`use_post_norm`, and `use_conv`.

---

## Optimizers

- **DGD**: Delta Gradient Descent (paper’s memory update rule).
- **GGD**: Generalized Gradient Descent (self-referential values, Eq. 59–60).
- **GM**: Generalized Momentum with persistent memory.
- **ContextSteeredOptimizer**: wraps any `torch.optim` optimizer and applies
  associative-memory preconditioning (`none`, `adagrad`, `adam`, `outer`).

---

## API Reference

### `HOPEModel`

```python
HOPEModel(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    frequencies: list[int] | None,
    cms_variant: str = "nested",  # nested | sequential | headwise | chain
    self_mod_depth: int = 2,
    heads: int = 4,
    self_mod_query_static: bool = False,
    self_mod_projection_mask: tuple[bool, bool, bool, bool, bool] | None = None,
    backbone: str = "titans",  # titans | attention
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
)
```

- `forward(x, time, update_memory=True)`
- `update_chunk(x, chunk_size=None, memory_chunk_size=None)`
- `self_update_from_logits()`

---

## Optimizer State Persistence (Continual Learning)

```python
from nested_learning.torch_optim import save_optimizer_state, load_optimizer_state

save_optimizer_state(optimizer, "optim_state.pt")
load_optimizer_state(optimizer, "optim_state.pt")
```

---

## Proof of Concept (Digits)

Example run (as in the paper’s continual-learning style setup):

```
Task A accuracy before: 0.287
Task B accuracy: 0.228
Task A accuracy after: 0.287
Forgetting: 0.000
```
* Accuracy is low because HOPE is not designed for vision task and this current implementation only uses a 128 dim, no CNN
* and this was ran on only 200 max samples.
---

## Package Layout

```
nested_learning/
  hope.py            # HOPE architecture
  nn.py              # Titans, Attention, AssociativeMemory, ContextFlow
  memory.py          # CMS variants
  torch_optim.py     # DGD / GGD / GM + steered optimizer
  kernels/
    cpu.py           # Numba CPU kernels
    gpu.py           # Triton GPU kernels
examples/
  continual_digits.py
```

---

## Credits

```
- NestedLearning.pdf (Ali Behrouz et al, and other Google Researcher) = Paper
- OpenAI Codex (Model=GPT-5.2 Codex Max) = Codebase
- @WindOfNature = Idea
```

---

## License

Apache-2.0
