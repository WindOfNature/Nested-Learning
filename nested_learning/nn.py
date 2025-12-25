"""Neural modules with nested-learning inspired updates."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from .tensor import Tensor
from .kernels import cpu as cpu_kernels


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        for attr in self.__dict__.values():
            if isinstance(attr, Tensor):
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                    elif isinstance(item, Tensor):
                        params.append(item)
        return params

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, seed: Optional[int] = None):
        super().__init__()
        scale = np.sqrt(2.0 / in_features)
        self.weight = Tensor.randn((in_features, out_features), requires_grad=True, seed=seed, name="weight")
        self.weight.data *= scale
        self.bias = Tensor.zeros((out_features,), requires_grad=True, name="bias") if bias else None

    def forward(self, x: Tensor) -> Tensor:
        if x.data.ndim == 1:
            x = Tensor(x.data[None, :], requires_grad=x.requires_grad)
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class LayerNorm(Module):
    def __init__(self, features: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = Tensor.ones((features,), requires_grad=True, name="gamma")
        self.beta = Tensor.zeros((features,), requires_grad=True, name="beta")
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) * (x - mean)).mean(axis=-1, keepdims=True)
        norm = (x - mean) / (var + self.eps).sqrt()
        return norm * self.gamma + self.beta


class MLP(Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, seed: Optional[int] = None):
        super().__init__()
        self.fc1 = Linear(in_features, hidden_features, seed=seed)
        self.fc2 = Linear(hidden_features, out_features, seed=seed)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x).relu())


class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class SelfModifyingLayer(Module):
    """Layer with learned update rule for self-modification."""

    def __init__(self, features: int, update_hidden: int = 64):
        super().__init__()
        self.base = Linear(features, features)
        self.rule = MLP(features * 2, update_hidden, features)
        self.scale = Tensor.ones((features,), requires_grad=True, name="selfmod_scale")

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x).relu()

    def self_update(self, x: Tensor, grad: Tensor):
        context = Tensor(np.concatenate([x.data, grad.data], axis=-1), requires_grad=False)
        delta = self.rule(context).tanh()
        self.base.weight.data += self.scale.data * delta.data.mean(axis=0)


class SelfReferentialTitan(Module):
    """Self-referential module with an inner optimizer for its update rule."""

    def __init__(self, features: int, update_hidden: int = 64, inner_lr: float = 1e-3):
        super().__init__()
        self.core = SelfModifyingLayer(features, update_hidden=update_hidden)
        self.rule_optimizer = None
        self.inner_lr = inner_lr

    def _init_inner_optimizer(self):
        if self.rule_optimizer is None:
            from .optim import AdamW
            self.rule_optimizer = AdamW(self.core.rule.parameters(), lr=self.inner_lr, weight_decay=1e-4)

    def forward(self, x: Tensor) -> Tensor:
        return self.core.forward(x)

    def self_update(self, x: Tensor, grad: Tensor):
        self.core.self_update(x, grad)
        self._init_inner_optimizer()
        target = Tensor(-grad.data, requires_grad=False)
        context = Tensor(np.concatenate([x.data, grad.data], axis=-1), requires_grad=False)
        pred = self.core.rule(context)
        loss = ((pred - target) * (pred - target)).mean()
        self.rule_optimizer.zero_grad()
        loss.backward()
        self.rule_optimizer.step()


class SelfModifyingStack(Module):
    """Stacked self-modifying titans for deeper self-referential updates."""

    def __init__(self, features: int, depth: int = 2, update_hidden: int = 64, inner_lr: float = 1e-3):
        super().__init__()
        self.layers = [SelfReferentialTitan(features, update_hidden=update_hidden, inner_lr=inner_lr) for _ in range(depth)]

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            out = layer(out).relu()
        return out

    def self_update(self, x: Tensor, grad: Tensor):
        for layer in self.layers:
            layer.self_update(x, grad)


class AdaptiveLinear(Module):
    """Linear layer that supports custom kernel execution."""

    def __init__(self, in_features: int, out_features: int, use_cpu_kernel: bool = True, seed: Optional[int] = None):
        super().__init__()
        self.use_cpu_kernel = use_cpu_kernel
        self.weight = Tensor.randn((in_features, out_features), requires_grad=True, seed=seed, name="weight")
        self.bias = Tensor.zeros((out_features,), requires_grad=True, name="bias")

    def forward(self, x: Tensor) -> Tensor:
        if self.use_cpu_kernel:
            out_data = cpu_kernels.matmul(x.data, self.weight.data) + self.bias.data
            return Tensor(out_data, requires_grad=x.requires_grad or self.weight.requires_grad)
        out = x @ self.weight
        return out + self.bias
