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
            # We check if it returns tuple
            if isinstance(output, tuple) and len(output) >= 4:
                eta = output[2]
                alpha = output[3]

                if isinstance(eta, torch.Tensor):
                    print(f"[{name}] Eta: Mean={eta.mean().item():.4f} Min={eta.min().item():.4f} Max={eta.max().item():.4f}")
                if isinstance(alpha, torch.Tensor):
                    print(f"[{name}] Alpha: Mean={alpha.mean().item():.4f}")

        return hook

    # Attach to first memory block
    if hasattr(model.cms, 'blocks'):
        model.cms.blocks[0].register_forward_hook(log_stats("CMS_Block_0"))

    # Run loop
    x = torch.randn(32, 64)
    y = torch.randint(0, 10, (32,))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("Running Debug Loop...")
    for i in range(5):
        print(f"--- Step {i} ---")
        logits = model(x, time=i)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Trigger update_chunk to check internal dynamics
        # update_chunk internally calls cms.forward(update=True) which calls MemoryBlock.update
        # We also want to check gradients on Encoder
        if model.encoder.weight.grad is not None:
             print(f"[Encoder] Grad Norm: {model.encoder.weight.grad.norm().item():.4f}")

        model.update_chunk(x)

if __name__ == "__main__":
    debug_hooks()
