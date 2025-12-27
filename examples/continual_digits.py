"""Continual learning evaluation on scikit-learn digits dataset (torch)."""

from __future__ import annotations

import argparse

import numpy as np
import time
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from nested_learning.hope import HOPEModel, HopeState
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
        if args.replay_ratio == 0.0:
            args.replay_ratio = auto_config.get("replay_ratio", args.replay_ratio)
        if args.memory_decay == 0.0:
            args.memory_decay = auto_config.get("memory_decay", args.memory_decay)
        if args.replay_ratio > 0.0:
            args.replay_weight = 0.5
        if args.task_b_epochs is None:
            args.task_b_epochs = max(args.epochs, 10)

    if args.preset == "adaptive":
        if args.cms_frequencies is None:
            if dataset_size <= 1000:
                args.cms_frequencies = [1, 2, 4, 8]
            else:
                args.cms_frequencies = [1, 8, 16, 32, 64]
        cms_config = HOPEModel.auto_scale_cms(
            dataset_size,
            task_count,
            backbone=args.backbone,
            batch_size=args.batch_size,
        )
        if args.cms_chunk_size is None:
            args.cms_chunk_size = cms_config.get("cms_chunk_size", args.cms_chunk_size)
        if args.cms_memory_chunk_size is None:
            args.cms_memory_chunk_size = cms_config.get("cms_memory_chunk_size", args.cms_memory_chunk_size)

    return args


def train_task(
    model: HOPEModel,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    batch_size: int,
    device: torch.device,
    rng: torch.Generator,
    cms_chunk_size: int,
    cms_memory_chunk_size: int,
    task_id: int,
    replay_weight: float,
    replay_ratio: float,
    dataset_size: int,
    task_count: int,
    batch_size_value: int,
    dynamic_cms: bool,
):
    def detach_state(state):
        if state is None:
            return None
        if isinstance(state, torch.Tensor):
            return state.detach()
        if isinstance(state, (list, tuple)):
            detached = [detach_state(item) for item in state]
            return type(state)(detached)
        return state

    model.train()
    global_step = 0
    for epoch in range(epochs):
        running_state = None
        indices = torch.randperm(x.size(0), generator=rng, device=x.device)
        for step in range(0, len(indices), batch_size):
            global_step += 1
            batch_idx = indices[step : step + batch_size]
            batch_x = x[batch_idx]
            batch_y = y[batch_idx]

            # 1. Remember new data
            model.remember(batch_x, batch_y, task_id=task_id)

            # 2. Sample Replay
            replay_data = None
            if replay_ratio > 0.0 and replay_weight > 0.0:
                replay_data = model.sample_replay(batch_size // 2)

            # 3. Train with task-weighted replay
            if replay_data is not None:
                rx, ry, rtask = replay_data
                combined_x = torch.cat([batch_x, rx], dim=0)
                combined_features, new_cms_states, _ = model.forward_features(
                    combined_x,
                    time=global_step,
                    update_memory=False,
                    state=running_state,
                )
                current_features = combined_features[: batch_x.size(0)]
                replay_features = combined_features[batch_x.size(0) :]
                logits_current, new_dec_state, _ = model.forward_decoder(
                    current_features,
                    task_id=task_id,
                    update_memory=False,
                    state=running_state,
                )
                loss_current = torch.nn.functional.cross_entropy(logits_current, batch_y)
                loss_replay = 0.0
                for task_value in torch.unique(rtask):
                    mask = rtask == task_value
                    if not mask.any():
                        continue
                    logits_replay, _, _ = model.forward_decoder(
                        replay_features[mask],
                        task_id=int(task_value.item()),
                        update_memory=False,
                        state=running_state,
                    )
                    loss_replay = loss_replay + torch.nn.functional.cross_entropy(logits_replay, ry[mask])
                loss = loss_current + replay_weight * loss_replay
            else:
                current_features, new_cms_states, _ = model.forward_features(
                    batch_x,
                    time=global_step,
                    update_memory=False,
                    state=running_state,
                )
                logits_current, new_dec_state, _ = model.forward_decoder(
                    current_features,
                    task_id=task_id,
                    update_memory=False,
                    state=running_state,
                )
                loss = torch.nn.functional.cross_entropy(logits_current, batch_y)
            running_state = HopeState(
                time=global_step,
                memory=detach_state(new_cms_states),
                decoder_memory=new_dec_state.detach(),
            )

            batch_acc = (logits_current.argmax(dim=-1) == batch_y).float().mean().item()
            if dynamic_cms:
                model.maybe_rescale_cms(
                    loss=loss.item(),
                    accuracy=batch_acc,
                    dataset_size=dataset_size,
                    task_count=task_count,
                    batch_size=batch_size_value,
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                maintenance_x = combined_x if replay_data is not None else batch_x
                _, maintenance_states, _ = model.forward_features(
                    maintenance_x,
                    time=global_step,
                    update_memory=True,
                    state=running_state,
                )
                if maintenance_states is not None:
                    running_state = HopeState(
                        time=global_step,
                        memory=detach_state(maintenance_states),
                        decoder_memory=running_state.decoder_memory if running_state else None,
                    )

            with torch.no_grad():
                if replay_data is not None:
                    model.forward_features(
                        combined_x,
                        time=global_step,
                        update_memory=True,
                        state=running_state,
                    )
                else:
                    model.forward_features(
                        batch_x,
                        time=global_step,
                        update_memory=True,
                        state=running_state,
                    )

            if epoch == epochs - 1 and step % (batch_size * 4) == 0:
                model.self_update_from_logits()


def evaluate(
    model: HOPEModel,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: torch.device,
    task_id: int,
) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for step in range(0, x.size(0), batch_size):
            batch_x = x[step : step + batch_size]
            batch_y = y[step : step + batch_size]
            logits = model.forward(batch_x, time=step, update_memory=False, task_id=task_id)
            preds = logits.argmax(dim=-1)
            correct += int((preds == batch_y).sum().item())
    return correct / x.size(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--task-a-epochs", type=int, default=None)
    parser.add_argument("--task-b-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
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

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = torch.Generator(device=device).manual_seed(args.seed)

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

    if args.cms_frequencies is None or args.cms_depth is None or args.cms_memory_chunk_size is None:
        raise ValueError("CMS frequencies, depth, and memory chunk size are required")
    if args.cms_chunk_size is None and not (args.preset == "adaptive" and args.auto_scale):
        raise ValueError("CMS chunk size is required unless using preset='adaptive' with auto-scale")

    xa_train = torch.tensor(xa_train, device=device)
    ya_train = torch.tensor(ya_train, device=device)
    xa_test = torch.tensor(xa_test, device=device)
    ya_test = torch.tensor(ya_test, device=device)
    xb_train = torch.tensor(xb_train, device=device)
    yb_train = torch.tensor(yb_train, device=device)
    xb_test = torch.tensor(xb_test, device=device)
    yb_test = torch.tensor(yb_test, device=device)

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
    if model.cms_chunk_size is None:
        model.cms_chunk_size = args.cms_chunk_size
    if model.cms_memory_chunk_size is None:
        model.cms_memory_chunk_size = args.cms_memory_chunk_size

    fast_param_ids = set()
    for module in model.modules():
        if hasattr(module, "fast_params"):
            fast_param_ids.update(id(param) for param in module.fast_params)
    slow_params = [param for param in model.parameters() if id(param) not in fast_param_ids]
    base_optimizer = torch.optim.AdamW
    if args.steered_optim:
        config = SteeredOptimizerConfig(precondition=args.precondition, weight_decay=1e-3)
        optimizer = ContextSteeredOptimizer(slow_params, base_optimizer, config=config, lr=1e-3)
    else:
        optimizer = base_optimizer(slow_params, lr=1e-3, weight_decay=1e-3)

    task_a_epochs = args.task_a_epochs if args.task_a_epochs is not None else args.epochs
    task_b_epochs = args.task_b_epochs if args.task_b_epochs is not None else args.epochs
    dynamic_cms = args.preset == "adaptive" and args.auto_scale

    train_start = time.perf_counter()
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
        dataset_size=total_train,
        task_count=2,
        batch_size_value=args.batch_size,
        dynamic_cms=dynamic_cms,
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
        dataset_size=total_train,
        task_count=2,
        batch_size_value=args.batch_size,
        dynamic_cms=dynamic_cms,
    )
    train_end = time.perf_counter()
    acc_b = evaluate(model, xb_test, yb_test, batch_size=args.batch_size, device=device, task_id=1)
    acc_a_after = evaluate(model, xa_test, ya_test, batch_size=args.batch_size, device=device, task_id=0)

    print(f"Task A accuracy before: {acc_a_before:.3f}")
    print(f"Task B accuracy: {acc_b:.3f}")
    print(f"Task A accuracy after: {acc_a_after:.3f}")
    print(f"Forgetting: {acc_a_before - acc_a_after:.3f}")
    print(f"Total training time: {train_end - train_start:.2f}s")


if __name__ == "__main__":
    main()
