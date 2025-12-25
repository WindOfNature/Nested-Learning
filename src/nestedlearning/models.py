"""Self-modifying Titans (minimal skeleton)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from nestedlearning.tensor import SimpleTensor, mse_loss


@dataclass
class SelfModifyingTitan:
    """A minimal self-modifying sequence model."""

    input_dim: int
    hidden_dim: int
    output_dim: int
    W1: SimpleTensor
    b1: SimpleTensor
    W2: SimpleTensor
    b2: SimpleTensor

    @classmethod
    def init(cls, input_dim: int, hidden_dim: int, output_dim: int) -> "SelfModifyingTitan":
        rng = np.random.default_rng(0)
        W1 = SimpleTensor(rng.standard_normal((input_dim, hidden_dim)) * 0.01)
        b1 = SimpleTensor(np.zeros((1, hidden_dim)))
        W2 = SimpleTensor(rng.standard_normal((hidden_dim, output_dim)) * 0.01)
        b2 = SimpleTensor(np.zeros((1, output_dim)))
        return cls(input_dim, hidden_dim, output_dim, W1, b1, W2, b2)

    def parameters(self) -> Iterable[SimpleTensor]:
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, inputs: SimpleTensor) -> SimpleTensor:
        hidden = (inputs @ self.W1 + self.b1).relu()
        return hidden @ self.W2 + self.b2

    def loss(self, inputs: SimpleTensor, targets: SimpleTensor) -> SimpleTensor:
        preds = self.forward(inputs)
        return mse_loss(preds, targets)

    def self_modify(self, loss: SimpleTensor, optimizer: object) -> None:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
