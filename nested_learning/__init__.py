"""Nested Learning package."""

from .nn import Linear, LayerNorm, MLP, NestedContextFlow, SelfReferentialTitan, SelfModifyingStack
from .torch_optim import ContextSteeredOptimizer, DGD, SteeredOptimizerConfig
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
    "NestedContextFlow",
    "DGD",
    "ContextSteeredOptimizer",
    "SteeredOptimizerConfig",
    "ContinuumMemorySystem",
    "NestedContinuumMemorySystem",
    "SequentialContinuumMemorySystem",
    "HeadwiseContinuumMemorySystem",
    "HOPEModel",
]
