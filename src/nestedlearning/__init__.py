"""Nested Learning package skeleton."""

from nestedlearning.core import HopeTrainer, NestedModule, NestedOptimizer, SelfModifyingModel
from nestedlearning.kernels import GLOBAL_KERNELS, KernelRegistry
from nestedlearning.memory import ContinuumMemorySystem
from nestedlearning.models import SelfModifyingTitan
from nestedlearning.optimizers import Adam, AdamW, Muon, Optimizer
from nestedlearning.tensor import SimpleTensor, mse_loss
from nestedlearning.version import __version__

__all__ = [
    "__version__",
    "HopeTrainer",
    "NestedModule",
    "NestedOptimizer",
    "SelfModifyingModel",
    "KernelRegistry",
    "GLOBAL_KERNELS",
    "ContinuumMemorySystem",
    "SelfModifyingTitan",
    "Optimizer",
    "Adam",
    "AdamW",
    "Muon",
    "SimpleTensor",
    "mse_loss",
]
