# Nested Learning
This repository provides an early skeleton for a PyPI-ready library that will
implement the Nested Learning paradigm and the HOPE (self-modifying) continual
learning system described in "Nested Learning: The Illusion of Deep Learning
Architecture" by Ali Behrouz et al. The initial focus is a minimal Continuum
Memory System (CMS), a self-modifying Titan-style model, and custom optimizers
without relying on PyTorch. The kernel registry currently defaults to NumPy CPU
kernels and is intended to be extended with custom Triton GPU kernels.

## Install (skeleton)
```bash
pip install -e .
```

## Quick start (skeleton)
```python
import numpy as np

from nestedlearning import Adam, ContinuumMemorySystem, HopeTrainer, SelfModifyingTitan, SimpleTensor

model = SelfModifyingTitan.init(input_dim=4, hidden_dim=8, output_dim=2)
optimizer = Adam(model.parameters(), lr=1e-3)
memory = ContinuumMemorySystem(short_term_capacity=32)

trainer = HopeTrainer(model=model, optimizer=optimizer, memory=memory)
inputs = SimpleTensor(np.ones((1, 4)))
targets = SimpleTensor(np.zeros((1, 2)))
trainer.continual_update((inputs, targets))
```
