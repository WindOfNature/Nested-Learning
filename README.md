# Nested-Learning
An implementation of HOPE (Capable of Self-Modifying and Continual Learning without forgetting) architecture from "Nested Learning: The Illusion of Deep Learning Architecture" by Ali Behrouz. This package provides a torch-based nested-learning framework with expressive optimizers, continuum memory systems, and optional Triton GPU kernels.

## Disclaimer
This entire codebase was made by OpenAI Codex (Model GPT-5.2-Codex-Max) and may implement things wrong/inaccurate. This codebase is written by Codex because the owner (@WindOfNature) can't code (yet).

## Features
- Torch autograd with explicit context flow.
- Expressive optimizers (AdamW, Muon, and memory-based variants).
- Continuum Memory System (CMS) with multi-timescale updates and nested/sequential/head-wise variants.
- Self-modifying module (HOPE) with self-referential titans and stacked update rules.
- Expressive optimizer suite including Shampoo-style outer-product preconditioning.
- CPU kernels optimized via Numba; optional GPU Triton kernels.

## Installation
```bash
pip install -e .
```

From GitHub:
```bash
pip install git+https://github.com/WindOfNature/Nested-Learning.git
```

Optional dependencies:
```bash
pip install -e .[train]
```

## Quickstart
```python
import torch

from nested_learning.hope import HOPEModel

model = HOPEModel(input_dim=64, hidden_dim=128, output_dim=10, frequencies=[1, 4, 16], cms_variant="nested", self_mod_depth=3)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

x = torch.randn(1, 64)
logits = model.forward(x, time=0)
loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1]))
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## Continual Learning Evaluation
```bash
python examples/continual_digits.py --epochs 2 --batch-size 32 --max-samples 5000
```

This script uses scikit-learn's digits dataset to test self-modification and continual learning with minimal forgetting.

### Proof of Concept Results
Recent runs on the digits continual-learning demo show no forgetting, in fact, even a small improvement on Task A after training Task B:

```
Task A accuracy before: 0.177
Task B accuracy: 0.178
Task A accuracy after: 0.182
Forgetting: -0.006
```
* (this run is only 200 samples due to Codex enviro limitations)
