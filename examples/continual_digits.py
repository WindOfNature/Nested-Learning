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


def apply_presets(args: argparse.Namespace, dataset_size: int, task_count: int) -> argparse.Namespace:
    if args.preset != "custom":
        preset_config = HOPEModel.preset_config(args.preset)
        args.memory_decay = preset_config["memory_decay"]
        args.replay_ratio = preset_config["replay_ratio"]
        args.replay_steps = preset_config["replay_steps"]
        args.self_mod_depth = preset_config["self_mod_depth"]
        args.nested_depth = preset_config["nested_depth"]
        args.nested_hidden = preset_config["nested_hidden"]

    if args.auto_scale:
        auto_config = HOPEModel.auto_scale_config(dataset_size, task_count)
        args.replay_buffer = auto_config.get("replay_buffer", args.replay_buffer)
        if args.replay_ratio > 0.0:
            args.replay_ratio = auto_config.get("replay_ratio", args.replay_ratio)
        args.memory_decay = auto_config.get("memory_decay", args.memory_decay)
        if args.replay_weight > 0.0:
            args.replay_weight = min(0.3, 0.05 + 0.02 * task_count)
        if args.task_b_epochs is None:
            args.task_b_epochs = max(args.epochs, 10)

    if args.preset == "adaptive":
        cms_config = HOPEModel.auto_scale_cms(
            dataset_size,
            task_count,
            backbone=args.backbone,
            batch_size=args.batch_size,
        )
        if args.cms_frequencies is None:
            args.cms_frequencies = cms_config.get("frequencies", args.cms_frequencies)
        if args.cms_depth is None:
            args.cms_depth = cms_config.get("cms_depth", args.cms_depth)
        if args.cms_chunk_size is None:
            args.cms_chunk_size = cms_config.get("cms_chunk_size", args.cms_chunk_size)
        if args.cms_memory_chunk_size is None:
            args.cms_memory_chunk_size = cms_config.get("cms_memory_chunk_size", args.cms_memory_chunk_size)

    return args


def train_task(
    model: HOPEModel,
    optimizer: torch.optim.Optimizer,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
    cms_chunk_size: int,
    cms_memory_chunk_size: int,
    task_id: int,
    replay_weight: float,
    replay_ratio: float,
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
            replay_data = None
            if replay_ratio > 0.0 and replay_weight > 0.0:
                replay_data = model.sample_replay(batch_size // 2)

            # 3. Train with task-weighted replay
            if replay_data is not None:
                rx, ry, rtask = replay_data
                logits_current = model.forward(batch_x, time=global_step, task_id=task_id)
                loss_current = torch.nn.functional.cross_entropy(logits_current, batch_y)
                loss_replay = 0.0
                for task_value in torch.unique(rtask):
                    mask = rtask == task_value
                    if not mask.any():
                        continue
                    logits_replay = model.forward(
                        rx[mask],
                        time=global_step,
                        task_id=int(task_value.item()),
                        update_memory=False,
                        detach_encoder=True,
                    )
                    loss_replay = loss_replay + torch.nn.functional.cross_entropy(logits_replay, ry[mask])
                loss = loss_current + replay_weight * loss_replay
                combined_x = torch.cat([batch_x, rx], dim=0)
            else:
                logits = model.forward(batch_x, time=global_step, task_id=task_id)
                loss = torch.nn.functional.cross_entropy(logits, batch_y)
                combined_x = batch_x
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # 5. Update Chunk (CMS Maintenance on Mixed Data)
            # We use the combined batch for chunk updates to ensure consistency
            chunk_buffer.append(combined_x)
            if (step // batch_size + 1) % cms_chunk_size == 0:
                chunk_x = torch.cat(chunk_buffer, dim=0)
                model.update_chunk(
                    chunk_x,
                    chunk_size=cms_chunk_size,
                    memory_chunk_size=cms_memory_chunk_size,
                    task_id=task_id,
                )
                chunk_buffer = []
            if epoch == epochs - 1 and step % (batch_size * 4) == 0:
                model.self_update_from_logits()
        if chunk_buffer:
            chunk_x = torch.cat(chunk_buffer, dim=0)
            model.update_chunk(
                chunk_x,
                chunk_size=cms_chunk_size,
                memory_chunk_size=cms_memory_chunk_size,
                task_id=task_id,
            )


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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--task-a-epochs", type=int, default=None)
    parser.add_argument("--task-b-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--cms-frequencies", type=str, default=None)
    parser.add_argument("--cms-depth", type=int, default=None)
    parser.add_argument("--cms-chunk-size", type=int, default=None)
    parser.add_argument("--cms-memory-chunk-size", type=int, default=None)
    parser.add_argument(
        "--preset",
        type=str,
        default="balanced",
        choices=["adaptive", "balanced", "fast_adapt", "high_retention", "custom"],
    )
    parser.add_argument("--auto-scale", dest="auto_scale", action="store_true", default=True)
    parser.add_argument("--no-auto-scale", dest="auto_scale", action="store_false")
    parser.add_argument("--cms-variant", type=str, default="nested", choices=["nested", "sequential", "headwise", "chain"])
    parser.add_argument("--nested-depth", type=int, default=2)
    parser.add_argument("--nested-hidden", type=int, default=128)
    parser.add_argument("--memory-decay", type=float, default=0.01)
    parser.add_argument("--replay-ratio", type=float, default=0.0)
    parser.add_argument("--replay-steps", type=int, default=1)
    parser.add_argument("--replay-buffer", type=int, default=2000)
    parser.add_argument("--replay-weight", type=float, default=0.0)
    parser.add_argument("--self-mod-depth", type=int, default=3)
    parser.add_argument("--self-mod-query-static", action="store_true")
    parser.add_argument("--backbone", type=str, default="titans", choices=["titans", "attention"])
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
    if args.cms_frequencies:
        args.cms_frequencies = [int(item.strip()) for item in args.cms_frequencies.split(",") if item.strip()]

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    (xa, ya), (xb, yb) = prepare_tasks()
    xa_train, xa_test, ya_train, ya_test = train_test_split(xa, ya, test_size=0.2, random_state=42)
    xb_train, xb_test, yb_train, yb_test = train_test_split(xb, yb, test_size=0.2, random_state=42)
    xa_train, ya_train = xa_train[: args.max_samples], ya_train[: args.max_samples]
    xb_train, yb_train = xb_train[: args.max_samples], yb_train[: args.max_samples]
    total_train = len(xa_train) + len(xb_train)

    projection_mask = (
        not args.freeze_k,
        not args.freeze_v,
        not args.freeze_q,
        not args.freeze_eta,
        not args.freeze_alpha,
    )

    args = apply_presets(args, dataset_size=total_train, task_count=2)

    if args.cms_frequencies is None or args.cms_depth is None or args.cms_chunk_size is None or args.cms_memory_chunk_size is None:
        raise ValueError("CMS hyperparameters are required unless using preset='adaptive'")

    if args.preset != "custom":
        model = HOPEModel.from_preset(
            input_dim=64,
            hidden_dim=128,
            output_dim=5,
            preset=args.preset,
            dataset_size=total_train,
            task_count=2,
            auto_scale=args.auto_scale,
            batch_size=args.batch_size,
            cms_variant="nested" if args.cms_variant == "chain" else args.cms_variant,
            frequencies=args.cms_frequencies,
            cms_depth=args.cms_depth,
            replay_steps=args.replay_steps,
            replay_buffer=args.replay_buffer,
            self_mod_query_static=args.self_mod_query_static,
            self_mod_projection_mask=projection_mask,
            backbone=args.backbone,
            heads=4,
        ).to(device)
    else:
        model = HOPEModel(
            input_dim=64,
            hidden_dim=128,
            output_dim=5,
            task_count=2,
            frequencies=args.cms_frequencies,
            cms_depth=args.cms_depth,
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
        ).to(device)
    base_optimizer = torch.optim.AdamW
    if args.steered_optim:
        config = SteeredOptimizerConfig(precondition=args.precondition, weight_decay=1e-3)
        optimizer = ContextSteeredOptimizer(model.parameters(), base_optimizer, config=config, lr=1e-3)
    else:
        optimizer = base_optimizer(model.parameters(), lr=1e-3, weight_decay=1e-3)

    task_a_epochs = args.task_a_epochs if args.task_a_epochs is not None else args.epochs
    task_b_epochs = args.task_b_epochs if args.task_b_epochs is not None else args.epochs

    train_task(
        model,
        optimizer,
        xa_train,
        ya_train,
        epochs=task_a_epochs,
        batch_size=args.batch_size,
        device=device,
        rng=rng,
        cms_chunk_size=args.cms_chunk_size,
        cms_memory_chunk_size=args.cms_memory_chunk_size,
        task_id=0,
        replay_weight=args.replay_weight,
        replay_ratio=args.replay_ratio,
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
        epochs=task_b_epochs,
        batch_size=args.batch_size,
        device=device,
        rng=rng,
        cms_chunk_size=args.cms_chunk_size,
        cms_memory_chunk_size=args.cms_memory_chunk_size,
        task_id=1,
        replay_weight=args.replay_weight,
        replay_ratio=args.replay_ratio,
    )
    acc_b = evaluate(model, xb_test, yb_test, batch_size=args.batch_size, device=device, task_id=1)
    acc_a_after = evaluate(model, xa_test, ya_test, batch_size=args.batch_size, device=device, task_id=0)

    print(f"Task A accuracy before: {acc_a_before:.3f}")
    print(f"Task B accuracy: {acc_b:.3f}")
    print(f"Task A accuracy after: {acc_a_after:.3f}")
    print(f"Forgetting: {acc_a_before - acc_a_after:.3f}")


if __name__ == "__main__":
    main()
