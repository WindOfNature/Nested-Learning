import torch
import numpy as np
import torch.nn.functional as F

from nested_learning.hope import HOPEModel
from nested_learning.nn import SelfReferentialTitan

def debug_hooks():
    model = HOPEModel(
        input_dim=64,
        hidden_dim=128,
        output_dim=10,
        frequencies=[1, 2, 4, 8],
        cms_variant="nested",
    )

    # Register hooks
    def log_stats(name):
        def hook(module, input, output):
            # For MemoryBlock, output is (out, hidden, eta, alpha, mem)
            if isinstance(output, tuple) and len(output) >= 4:
                out = output[0]
                eta = output[2]
                mem = output[4] if len(output) > 4 else None

                print(f"[{name}] Out Mean={out.mean().item():.4f} Std={out.std().item():.4f}")
                if isinstance(eta, torch.Tensor):
                    print(f"[{name}] Eta Mean={eta.mean().item():.4f}")
                if mem is not None:
                    print(f"[{name}] Mem Mean={mem.mean().item():.4f} Std={mem.std().item():.4f}")

        return hook

    def log_titan_signals(name, layer: SelfReferentialTitan, context: torch.Tensor):
        with torch.no_grad():
            x_ln = layer.norm(context)
            x_l2 = F.normalize(x_ln, p=2, dim=-1)
            signals = layer.generate_signals(x_l2)
            print(
                f"[{name}] eta={signals.eta.mean().item():.4f} "
                f"alpha={signals.alpha.mean().item():.4f} "
                f"q={signals.q.norm(dim=-1).mean().item():.4f} "
                f"k={signals.k.norm(dim=-1).mean().item():.4f} "
                f"v={signals.v.norm(dim=-1).mean().item():.4f} "
                f"mem={signals.memory.norm(dim=-1).mean().item():.4f}"
            )

    def snapshot_titan_params(layer: SelfReferentialTitan):
        params = {}
        for name, module in [
            ("mk", layer.mk),
            ("mv", layer.mv),
            ("mq", layer.mq),
            ("meta", layer.meta),
            ("malpha", layer.malpha),
            ("mmemory", layer.mmemory),
        ]:
            params[name] = [p.detach().clone() for p in module.parameters()]
        return params

    def delta_norms(before, layer: SelfReferentialTitan):
        deltas = {}
        for name, module in [
            ("mk", layer.mk),
            ("mv", layer.mv),
            ("mq", layer.mq),
            ("meta", layer.meta),
            ("malpha", layer.malpha),
            ("mmemory", layer.mmemory),
        ]:
            total = 0.0
            for idx, p in enumerate(module.parameters()):
                total += (p.detach() - before[name][idx]).norm().item()
            deltas[name] = total
        return deltas

    # Attach to first memory block
    if hasattr(model.cms, 'blocks'):
        for i, block in enumerate(model.cms.blocks):
            block.register_forward_hook(log_stats(f"CMS_Block_{i}"))

    if hasattr(model, 'decoder_mem'):
        model.decoder_mem.register_forward_hook(log_stats("Decoder_Mem"))

    # Run loop
    x = torch.randn(32, 64)
    y = torch.randint(0, 10, (32,))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("Running Forensic Debug Loop...")
    for i in range(5):
        print(f"--- Step {i} ---")
        logits = model(x, time=i, update_memory=False)

        # Explosion Detector
        if torch.isnan(logits).any():
            print("!!! NAN DETECTED IN LOGITS !!!")
            break

        loss = criterion(logits, y)
        if torch.isnan(loss):
            print("!!! NAN DETECTED IN LOSS !!!")
            break

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if model._last_context is not None and model._last_context.grad is not None:
            grad_norm = model._last_context.grad.norm(dim=-1).mean().item()
            print(f"[LossGrad] Context grad mean norm={grad_norm:.4f}")

        if model.self_mod is not None and model._last_context is not None:
            for idx, layer in enumerate(model.self_mod.layers):
                log_titan_signals(f"Titans_{idx}", layer, model._last_context.detach())

            before = [snapshot_titan_params(layer) for layer in model.self_mod.layers]
            model.self_update_from_logits()
            for idx, layer in enumerate(model.self_mod.layers):
                deltas = delta_norms(before[idx], layer)
                delta_str = " ".join([f"{name}={value:.4f}" for name, value in deltas.items()])
                print(f"[Titans_{idx}] update_deltas {delta_str}")

        optimizer.step()

        # Skip offline maintenance to keep debug graph free of in-place updates.

if __name__ == "__main__":
    debug_hooks()
