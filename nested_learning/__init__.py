"""Nested Learning package."""

from .nn import (
    AssociativeMemory,
    AttentionBlock,
    Linear,
    LayerNorm,
    MLP,
    NestedContextFlow,
    SelfReferentialTitan,
    SelfModifyingStack,
)
from .torch_optim import (
    ContextSteeredOptimizer,
    DGD,
    GGD,
    GM,
    SteeredOptimizerConfig,
    load_optimizer_state,
    save_optimizer_state,
)
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
    "AssociativeMemory",
    "AttentionBlock",
    "DGD",
    "GGD",
    "GM",
    "ContextSteeredOptimizer",
    "SteeredOptimizerConfig",
    "save_optimizer_state",
    "load_optimizer_state",
    "ContinuumMemorySystem",
    "NestedContinuumMemorySystem",
    "SequentialContinuumMemorySystem",
    "HeadwiseContinuumMemorySystem",
    "HOPEModel",
]
