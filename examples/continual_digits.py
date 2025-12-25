"""Continual learning evaluation on scikit-learn digits dataset (torch)."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from nested_learning.hope import HOPEModel
from nested_learning.torch_optim import ContextSteeredOptimizer, SteeredOptimizerConfig


def prepare_tasks():
    digits = load_digits()
    x = digits.data.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)
    task_a = (x[y < 5], y[y < 5])
    task_b = (x[y >= 5], y[y >= 5] - 5)
    return task_a, task_b


def train_task(
    model: HOPEModel,
    optimizer: torch.optim.Optimizer,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
    chunk_size: int,
    memory_chunk_size: int,
):
    model.train()
    for epoch in range(epochs):
        indices = rng.permutation(len(x))
        chunk_buffer = []
        for step in range(0, len(indices), batch_size):
            batch_idx = indices[step : step + batch_size]
            batch_x = torch.tensor(x[batch_idx], device=device)
            batch_y = torch.tensor(y[batch_idx], device=device)
            logits = model.forward(batch_x, time=epoch * len(x) + step)
            loss = torch.nn.functional.cross_entropy(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            chunk_buffer.append(batch_x)
            if (step // batch_size + 1) % chunk_size == 0:
                chunk_x = torch.cat(chunk_buffer, dim=0)
                model.update_chunk(chunk_x, chunk_size=chunk_size, memory_chunk_size=memory_chunk_size)
                chunk_buffer = []
            if epoch == epochs - 1 and step % (batch_size * 4) == 0:
                model.self_update_from_logits()
        if chunk_buffer:
            chunk_x = torch.cat(chunk_buffer, dim=0)
            model.update_chunk(chunk_x, chunk_size=chunk_size, memory_chunk_size=memory_chunk_size)


def evaluate(model: HOPEModel, x: np.ndarray, y: np.ndarray, batch_size: int, device: torch.device) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for step in range(0, len(x), batch_size):
            batch_x = torch.tensor(x[step : step + batch_size], device=device)
            batch_y = torch.tensor(y[step : step + batch_size], device=device)
            logits = model.forward(batch_x, time=step, update_memory=False)
            preds = logits.argmax(dim=-1)
            correct += int((preds == batch_y).sum().item())
    return correct / len(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--memory-chunk-size", type=int, default=4)
    parser.add_argument("--cms-variant", type=str, default="nested", choices=["nested", "sequential", "headwise", "chain"])
    parser.add_argument("--nested-depth", type=int, default=2)
    parser.add_argument("--nested-hidden", type=int, default=128)
    parser.add_argument("--memory-decay", type=float, default=0.1)
    parser.add_argument("--replay-ratio", type=float, default=0.2)
    parser.add_argument("--replay-steps", type=int, default=1)
    parser.add_argument("--replay-buffer", type=int, default=128)
    parser.add_argument("--self-mod-depth", type=int, default=3)
    parser.add_argument("--self-mod-query-static", action="store_true")
    parser.add_argument("--steered-optim", action="store_true")
    parser.add_argument("--precondition", type=str, default="outer", choices=["none", "adagrad", "adam", "outer"])
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    (xa, ya), (xb, yb) = prepare_tasks()
    xa_train, xa_test, ya_train, ya_test = train_test_split(xa, ya, test_size=0.2, random_state=42)
    xb_train, xb_test, yb_train, yb_test = train_test_split(xb, yb, test_size=0.2, random_state=42)
    xa_train, ya_train = xa_train[: args.max_samples], ya_train[: args.max_samples]
    xb_train, yb_train = xb_train[: args.max_samples], yb_train[: args.max_samples]

    model = HOPEModel(
        input_dim=64,
        hidden_dim=128,
        output_dim=5,
        frequencies=[1, 5, 10],
        cms_variant="nested" if args.cms_variant == "chain" else args.cms_variant,
        self_mod_depth=args.self_mod_depth,
        heads=4,
        nested_depth=args.nested_depth,
        nested_hidden=args.nested_hidden,
        memory_decay=args.memory_decay,
        replay_ratio=args.replay_ratio,
        replay_steps=args.replay_steps,
        replay_buffer=args.replay_buffer,
        self_mod_query_static=args.self_mod_query_static,
    ).to(device)
    base_optimizer = torch.optim.AdamW
    if args.steered_optim:
        config = SteeredOptimizerConfig(precondition=args.precondition, weight_decay=1e-3)
        optimizer = ContextSteeredOptimizer(model.parameters(), base_optimizer, config=config, lr=1e-3)
    else:
        optimizer = base_optimizer(model.parameters(), lr=1e-3, weight_decay=1e-3)

    train_task(
        model,
        optimizer,
        xa_train,
        ya_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        rng=rng,
        chunk_size=args.chunk_size,
        memory_chunk_size=args.memory_chunk_size,
    )
    acc_a_before = evaluate(model, xa_test, ya_test, batch_size=args.batch_size, device=device)

    train_task(
        model,
        optimizer,
        xb_train,
        yb_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        rng=rng,
        chunk_size=args.chunk_size,
        memory_chunk_size=args.memory_chunk_size,
    )
    acc_b = evaluate(model, xb_test, yb_test, batch_size=args.batch_size, device=device)
    acc_a_after = evaluate(model, xa_test, ya_test, batch_size=args.batch_size, device=device)

    print(f"Task A accuracy before: {acc_a_before:.3f}")
    print(f"Task B accuracy: {acc_b:.3f}")
    print(f"Task A accuracy after: {acc_a_after:.3f}")
    print(f"Forgetting: {acc_a_before - acc_a_after:.3f}")


if __name__ == "__main__":
    main()
