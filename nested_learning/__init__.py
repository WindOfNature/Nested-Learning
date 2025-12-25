"""Nested Learning package."""

from .nn import Linear, LayerNorm, MLP, SelfReferentialTitan, SelfModifyingStack
from .torch_optim import DGD
from .memory import (
    ContinuumMemorySystem,
    NestedContinuumMemorySystem,
    SequentialContinuumMemorySystem,
    HeadwiseContinuumMemorySystem,
)
from .hope import HOPEModel

__all__ = [
    "Linear",
    "LayerNorm",
    "MLP",
    "SelfReferentialTitan",
    "SelfModifyingStack",
    "DGD",
    "ContinuumMemorySystem",
    "NestedContinuumMemorySystem",
    "SequentialContinuumMemorySystem",
    "HeadwiseContinuumMemorySystem",
    "HOPEModel",
]
