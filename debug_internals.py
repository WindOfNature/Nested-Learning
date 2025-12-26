import torch
import numpy as np
from nested_learning.hope import HOPEModel

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
        logits = model(x, time=i)

        # Explosion Detector
        if torch.isnan(logits).any():
            print("!!! NAN DETECTED IN LOGITS !!!")
            break

        loss = criterion(logits, y)
        if torch.isnan(loss):
            print("!!! NAN DETECTED IN LOSS !!!")
            break

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.update_chunk(x)

if __name__ == "__main__":
    debug_hooks()
