"""Continual learning evaluation on scikit-learn digits dataset."""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from nested_learning.hope import HOPEModel
from nested_learning.tensor import Tensor, cross_entropy
from nested_learning.optim import AdamW


def prepare_tasks():
    digits = load_digits()
    x = digits.data.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)
    task_a = (x[y < 5], y[y < 5])
    task_b = (x[y >= 5], y[y >= 5] - 5)
    return task_a, task_b


def train_task(
    model: HOPEModel,
    optimizer: AdamW,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    task_id: int,
    batch_size: int,
):
    for epoch in range(epochs):
        indices = np.random.permutation(len(x))
        for step in range(0, len(indices), batch_size):
            batch_idx = indices[step : step + batch_size]
            input_tensor = Tensor(x[batch_idx], requires_grad=True)
            logits = model.forward(input_tensor, time=epoch * len(x) + step)
            loss = cross_entropy(logits, y[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch == epochs - 1 and step % (batch_size * 4) == 0:
                model.self_update_from_logits()


def evaluate(model: HOPEModel, x: np.ndarray, y: np.ndarray, batch_size: int) -> float:
    correct = 0
    for step in range(0, len(x), batch_size):
        batch_x = x[step : step + batch_size]
        batch_y = y[step : step + batch_size]
        logits = model.forward(Tensor(batch_x, requires_grad=False), time=step)
        preds = np.argmax(logits.data, axis=-1)
        correct += int((preds == batch_y).sum())
    return correct / len(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()

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
        cms_variant="nested",
        self_mod_depth=3,
        heads=4,
    )
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    train_task(model, optimizer, xa_train, ya_train, epochs=args.epochs, task_id=0, batch_size=args.batch_size)
    acc_a_before = evaluate(model, xa_test, ya_test, batch_size=args.batch_size)

    train_task(model, optimizer, xb_train, yb_train, epochs=args.epochs, task_id=1, batch_size=args.batch_size)
    acc_b = evaluate(model, xb_test, yb_test, batch_size=args.batch_size)
    acc_a_after = evaluate(model, xa_test, ya_test, batch_size=args.batch_size)

    print(f"Task A accuracy before: {acc_a_before:.3f}")
    print(f"Task B accuracy: {acc_b:.3f}")
    print(f"Task A accuracy after: {acc_a_after:.3f}")
    print(f"Forgetting: {acc_a_before - acc_a_after:.3f}")


if __name__ == "__main__":
    main()
