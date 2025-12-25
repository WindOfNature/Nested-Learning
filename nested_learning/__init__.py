"""Nested Learning package."""

from .tensor import Tensor
from .nn import Linear, LayerNorm, MLP, SelfModifyingLayer, SelfReferentialTitan, SelfModifyingStack
from .torch_optim import DGD
from .memory import (
    ContinuumMemorySystem,
    NestedContinuumMemorySystem,
    SequentialContinuumMemorySystem,
    HeadwiseContinuumMemorySystem,
)
from .hope import HOPEModel

__all__ = [
    "Tensor",
    "Linear",
    "LayerNorm",
    "MLP",
    "SelfModifyingLayer",
    "SelfReferentialTitan",
    "SelfModifyingStack",
    "DGD",
    "ContinuumMemorySystem",
    "NestedContinuumMemorySystem",
    "SequentialContinuumMemorySystem",
    "HeadwiseContinuumMemorySystem",
    "HOPEModel",
]
