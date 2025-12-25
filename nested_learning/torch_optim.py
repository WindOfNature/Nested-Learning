"""Torch-based optimizers for Nested Learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import torch


class DGD(torch.optim.Optimizer):
    """Delta Gradient Descent (DGD) with L2-regression-inspired updates."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        beta: float = 0.9,
        alpha: float = 0.5,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, beta=beta, alpha=alpha, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, lr_override: float | None = None, weight_decay_override: float | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"] if lr_override is None else lr_override
            beta = group["beta"]
            alpha = group["alpha"]
            weight_decay = group["weight_decay"] if weight_decay_override is None else weight_decay_override
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if "memory" not in state:
                    state["memory"] = torch.zeros_like(param)
                grad = param.grad
                delta = grad + alpha * (grad - state["memory"])
                if weight_decay:
                    delta = delta + weight_decay * param
                param.add_(delta, alpha=-lr)
                state["memory"].mul_(beta).add_(grad, alpha=1 - beta)
        return loss


@dataclass
class SteeredOptimizerConfig:
    """Configuration for steering base torch optimizers with associative-memory state."""

    memory_beta: float = 0.9
    variance_beta: float = 0.999
    alpha: float = 0.5
    eps: float = 1e-8
    precondition: str = "adam"
    weight_decay: float | None = None


def _matrix_inv_sqrt(matrix: torch.Tensor, eps: float) -> torch.Tensor:
    if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square matrix for inverse sqrt")
    jitter = eps * torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    stabilized = torch.nan_to_num(matrix) + jitter
    try:
        eigvals, eigvecs = torch.linalg.eigh(stabilized)
        eigvals = torch.clamp(eigvals, min=eps)
        inv_sqrt = eigvecs @ torch.diag_embed(eigvals.rsqrt()) @ eigvecs.transpose(-1, -2)
        return inv_sqrt
    except RuntimeError:
        u, s, v = torch.linalg.svd(stabilized, full_matrices=False)
        s = torch.clamp(s, min=eps)
        return (u * s.rsqrt()) @ v


class ContextSteeredOptimizer:
    """Wrap a torch optimizer with NL-style associative memory steering."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        base_optimizer: Callable[..., torch.optim.Optimizer],
        config: SteeredOptimizerConfig | None = None,
        **base_kwargs,
    ):
        self.params = list(params)
        self.base_optimizer = base_optimizer(self.params, **base_kwargs)
        self.config = config or SteeredOptimizerConfig()
        self.state: dict[torch.nn.Parameter, dict[str, torch.Tensor]] = {}

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _get_state(self, param: torch.Tensor) -> dict[str, torch.Tensor]:
        if param not in self.state:
            self.state[param] = {
                "memory": torch.zeros_like(param),
                "variance": torch.zeros_like(param),
            }
            if param.dim() >= 2:
                dim0 = param.shape[0]
                dim1 = param.shape[1]
                self.state[param]["left"] = torch.zeros((dim0, dim0), device=param.device, dtype=param.dtype)
                self.state[param]["right"] = torch.zeros((dim1, dim1), device=param.device, dtype=param.dtype)
        return self.state[param]

    def _steer_grad(self, param: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        state = self._get_state(param)
        memory = state["memory"]
        variance = state["variance"]

        memory.mul_(cfg.memory_beta).add_(grad, alpha=1 - cfg.memory_beta)
        variance.mul_(cfg.variance_beta).addcmul_(grad, grad, value=1 - cfg.variance_beta)

        delta = grad + cfg.alpha * (grad - memory)

        if cfg.precondition == "none":
            steered = delta
        elif cfg.precondition == "adagrad":
            steered = delta / (variance.sqrt() + cfg.eps)
        elif cfg.precondition == "adam":
            steered = delta / (variance.sqrt() + cfg.eps)
        elif cfg.precondition == "outer":
            if param.dim() < 2:
                steered = delta / (variance.sqrt() + cfg.eps)
            else:
                left = state["left"]
                right = state["right"]
                left.mul_(cfg.variance_beta).add_(grad @ grad.transpose(-1, -2), alpha=1 - cfg.variance_beta)
                right.mul_(cfg.variance_beta).add_(grad.transpose(-1, -2) @ grad, alpha=1 - cfg.variance_beta)
                left_inv = _matrix_inv_sqrt(left, cfg.eps)
                right_inv = _matrix_inv_sqrt(right, cfg.eps)
                steered = left_inv @ grad @ right_inv
        else:
            raise ValueError(f"Unknown precondition mode: {cfg.precondition}")

        if cfg.weight_decay is not None:
            steered = steered + cfg.weight_decay * param
        return steered

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        original_grads: Mapping[torch.nn.Parameter, torch.Tensor] = {}
        for param in self.params:
            if param.grad is None:
                continue
            original_grads[param] = param.grad
            param.grad = self._steer_grad(param, param.grad)
        self.base_optimizer.step()
        for param, grad in original_grads.items():
            param.grad = grad
        return loss
