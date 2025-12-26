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

### A) HOPE (Titans backbone, preset defaults)

```python
import torch
from nested_learning.hope import HOPEModel

model = HOPEModel.from_preset(
    input_dim=64,
    hidden_dim=128,
    output_dim=10,
    preset="balanced",
    dataset_size=10_000,
    task_count=4,
    frequencies=[1, 2, 4, 8],
    cms_depth=2,
    backbone="titans",
)

x = torch.randn(8, 64)
logits = model(x, time=0)
```

### B) Hope-Attention (preset defaults)

```python
import torch
from nested_learning.hope import HOPEModel

model = HOPEModel.from_preset(
    input_dim=64,
    hidden_dim=128,
    output_dim=10,
    preset="balanced",
    dataset_size=10_000,
    task_count=4,
    backbone="attention",
)

x = torch.randn(8, 64)
logits = model(x, time=0)
```

### C) Continual learning demo (digits)

```bash
PYTHONPATH=. python examples/continual_digits.py \
  --max-samples 500
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
    task_count: int | None = None,
    frequencies: list[int] | None,
    cms_depth: int = 2,
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
    replay_steps: int = 0,
    replay_buffer: int = 128,
    use_conv: bool = True,
    conv_kernel: int = 3,
    use_pre_norm: bool = True,
    use_post_norm: bool = True,
)
```

- `forward(x, time, update_memory=True, task_id=None, detach_encoder=False)`
- `update_chunk(x, chunk_size=None, memory_chunk_size=None, task_id=None)`
- `self_update_from_logits()`

### `HOPEModel.from_preset`

```python
HOPEModel.from_preset(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    preset: str = "balanced",  # balanced | fast_adapt | high_retention
    dataset_size: int | None = None,
    task_count: int | None = None,
    auto_scale: bool = True,
    **overrides,
)
```

The preset path reduces the number of knobs for most users. Advanced overrides
(e.g., `frequencies`, `replay_ratio`, `self_mod_depth`) are still available but
not required for a solid baseline.

### Presets & Auto-scale (recommended)

If you don’t want to tune hyperparameters, use `preset="balanced"` with
`auto_scale=True`. Auto-scaling adapts replay buffer size, replay ratio, memory
decay, and (for Titans) CMS frequencies/depth/chunk sizes based on dataset size
and task count.

> Note: Presets default to **no replay**. Replay is only active if you pass
> non-zero `replay_ratio` and `replay_weight` in your training code.

### CMS Hyperparameters (auto-scaled)

`frequencies` defines the CMS update rates (Eq. 70–71 in the paper). `cms_depth`
controls the per-level MLP depth inside each memory block. By default, HOPE
auto-scales CMS **frequencies/depth/chunk sizes** when using the Titans
backbone, and uses a fixed schedule when using Hope-Attention. You can still
override any of them via `--cms-*` flags or constructor arguments.

`hope_levels` + `lowest_frequency` can still be used as a convenience to generate
frequencies, but presets no longer require explicit CMS configuration.
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
Task A accuracy before: 0.740
Task B accuracy: 0.933
Task A accuracy after: 0.785
Forgetting: -0.044
```
* Accuracy is shown for the default preset + auto-scale settings using 500 max samples (replay disabled).
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
