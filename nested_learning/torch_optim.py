"""Torch-based optimizers for Nested Learning."""

from __future__ import annotations

from typing import Iterable

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
