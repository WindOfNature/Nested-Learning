"""Continual learning evaluation on scikit-learn digits dataset (torch)."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from nested_learning.hope import HOPEModel
from nested_learning.torch_optim import (
    ContextSteeredOptimizer,
    SteeredOptimizerConfig,
    load_optimizer_state,
    save_optimizer_state,
)


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
    task_id: int,
):
    model.train()
    global_step = 0
    for epoch in range(epochs):
        indices = rng.permutation(len(x))
        chunk_buffer = []
        for step in range(0, len(indices), batch_size):
            global_step += 1
            batch_idx = indices[step : step + batch_size]
            batch_x = torch.tensor(x[batch_idx], device=device)
            batch_y = torch.tensor(y[batch_idx], device=device)

            # 1. Remember new data
            model.remember(batch_x, batch_y, task_id=task_id)

            # 2. Sample Replay
            replay_data = model.sample_replay(batch_size // 2)

            # 3. Mix
            if replay_data is not None:
                rx, ry = replay_data
                combined_x = torch.cat([batch_x, rx], dim=0)
                combined_y = torch.cat([batch_y, ry], dim=0)
            else:
                combined_x = batch_x
                combined_y = batch_y

            # 4. Train on Combined Batch (Encoder, Decoder, Memory all learn from mixed)
            logits = model.forward(combined_x, time=global_step, task_id=task_id)
            loss = torch.nn.functional.cross_entropy(logits, combined_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # 5. Update Chunk (CMS Maintenance on Mixed Data)
            # We use the combined batch for chunk updates to ensure consistency
            chunk_buffer.append(combined_x)
            if (step // batch_size + 1) % chunk_size == 0:
                chunk_x = torch.cat(chunk_buffer, dim=0)
                model.update_chunk(chunk_x, chunk_size=chunk_size, memory_chunk_size=memory_chunk_size, task_id=task_id)
                chunk_buffer = []
            if epoch == epochs - 1 and step % (batch_size * 4) == 0:
                model.self_update_from_logits()
        if chunk_buffer:
            chunk_x = torch.cat(chunk_buffer, dim=0)
            model.update_chunk(chunk_x, chunk_size=chunk_size, memory_chunk_size=memory_chunk_size, task_id=task_id)


def evaluate(
    model: HOPEModel,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
    task_id: int,
) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for step in range(0, len(x), batch_size):
            batch_x = torch.tensor(x[step : step + batch_size], device=device)
            batch_y = torch.tensor(y[step : step + batch_size], device=device)
            logits = model.forward(batch_x, time=step, update_memory=False, task_id=task_id)
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
    parser.add_argument("--memory-decay", type=float, default=0.01)
    parser.add_argument("--replay-ratio", type=float, default=0.5)
    parser.add_argument("--replay-steps", type=int, default=1)
    parser.add_argument("--replay-buffer", type=int, default=2000)
    parser.add_argument("--self-mod-depth", type=int, default=3)
    parser.add_argument("--self-mod-query-static", action="store_true")
    parser.add_argument("--backbone", type=str, default="titans", choices=["titans", "attention"])
    parser.add_argument("--hope-levels", type=int, default=0)
    parser.add_argument("--lowest-frequency", type=int, default=1)
    parser.add_argument("--freeze-k", action="store_true")
    parser.add_argument("--freeze-v", action="store_true")
    parser.add_argument("--freeze-q", action="store_true")
    parser.add_argument("--freeze-eta", action="store_true")
    parser.add_argument("--freeze-alpha", action="store_true")
    parser.add_argument("--steered-optim", action="store_true")
    parser.add_argument("--precondition", type=str, default="outer", choices=["none", "adagrad", "adam", "outer"])
    parser.add_argument("--optimizer-state-path", type=str, default="")
    parser.add_argument("--reset-optimizer-between-tasks", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    (xa, ya), (xb, yb) = prepare_tasks()
    xa_train, xa_test, ya_train, ya_test = train_test_split(xa, ya, test_size=0.2, random_state=42)
    xb_train, xb_test, yb_train, yb_test = train_test_split(xb, yb, test_size=0.2, random_state=42)
    xa_train, ya_train = xa_train[: args.max_samples], ya_train[: args.max_samples]
    xb_train, yb_train = xb_train[: args.max_samples], yb_train[: args.max_samples]

    projection_mask = (
        not args.freeze_k,
        not args.freeze_v,
        not args.freeze_q,
        not args.freeze_eta,
        not args.freeze_alpha,
    )

    model = HOPEModel(
        input_dim=64,
        hidden_dim=128,
        output_dim=5,
        task_count=2,
        frequencies=None if args.hope_levels else [1, 2, 4, 8],
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
        self_mod_projection_mask=projection_mask,
        backbone=args.backbone,
        hope_levels=args.hope_levels or None,
        lowest_frequency=args.lowest_frequency,
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
        task_id=0,
    )
    acc_a_before = evaluate(model, xa_test, ya_test, batch_size=args.batch_size, device=device, task_id=0)

    if args.optimizer_state_path:
        save_optimizer_state(optimizer, args.optimizer_state_path)
    if args.reset_optimizer_between_tasks:
        if args.optimizer_state_path:
            load_optimizer_state(optimizer, args.optimizer_state_path)
        else:
            if args.steered_optim:
                config = SteeredOptimizerConfig(precondition=args.precondition, weight_decay=1e-3)
                optimizer = ContextSteeredOptimizer(model.parameters(), base_optimizer, config=config, lr=1e-3)
            else:
                optimizer = base_optimizer(model.parameters(), lr=1e-3, weight_decay=1e-3)

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
        task_id=1,
    )
    acc_b = evaluate(model, xb_test, yb_test, batch_size=args.batch_size, device=device, task_id=1)
    acc_a_after = evaluate(model, xa_test, ya_test, batch_size=args.batch_size, device=device, task_id=0)

    print(f"Task A accuracy before: {acc_a_before:.3f}")
    print(f"Task B accuracy: {acc_b:.3f}")
    print(f"Task A accuracy after: {acc_a_after:.3f}")
    print(f"Forgetting: {acc_a_before - acc_a_after:.3f}")


if __name__ == "__main__":
    main()
