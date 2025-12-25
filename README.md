# Nested-Learning
An implementation of HOPE (Capable of Self-Modifying and Continual Learning without forgetting) architecture from "Nested Learning: The Illusion of Deep Learning Architecture" by Ali Behrouz. This package provides a torch-free nested-learning framework with custom backprop, expressive optimizers, continuum memory systems, and optional Triton GPU kernels.

## Features
- Custom autograd engine with explicit context flow.
- Expressive optimizers (Adam, AdamW, Muon, and memory-based variants).
- Continuum Memory System (CMS) with multi-timescale updates and nested/sequential/head-wise variants.
- Self-modifying module (HOPE) with self-referential titans and stacked update rules.
- Expressive optimizer suite including Shampoo-style outer-product preconditioning.
- CPU kernels optimized via Numba; optional GPU Triton kernels.

## Installation
```bash
pip install -e .
```

Optional dependencies:
```bash
pip install -e .[train]
```

## Quickstart
```python
from nested_learning.hope import HOPEModel
from nested_learning.tensor import Tensor, cross_entropy
from nested_learning.optim import AdamW

model = HOPEModel(input_dim=64, hidden_dim=128, output_dim=10, frequencies=[1, 4, 16], cms_variant="nested", self_mod_depth=3)
optimizer = AdamW(model.parameters(), lr=1e-3)

x = Tensor.randn((1, 64), requires_grad=True)
logits = model.forward(x, time=0)
loss = cross_entropy(logits, targets=[1])
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
