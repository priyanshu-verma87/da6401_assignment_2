"""Reusable custom layers 
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """
    Inverted Dropout implemented from scratch.

    During TRAINING:
        1. Sample a binary mask ~ Bernoulli(1 - p) for each element
        2. Zero out elements where mask == 0
        3. Scale surviving elements by 1/(1-p): inverted dropout scaling
           This keeps the expected value of each activation unchanged, so no rescaling is needed at inference time.

    During EVAL (self.training == False):
        Pass input through unchanged - no masking, no scaling.

    Args:
        p (float): probability of dropping a unit. Default 0.5.

        
    Why inverted dropout?
        Inverted dropout scales at train time instead, so the forward pass at inference is just an identity - cleaner and more efficient.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()

        if not (0.0 <= p < 1.0):
            raise ValueError (f"Dropout probability must be in [0, 1), got {p}")
        
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Evaluation mode: identity pass 
        if not self.training:
            return x

        # p == 0: nothing to drop 
        if self.p == 0.0:
            return x

        # Build binary keep-mask 
        # Each element is kept with probability (1 - p)
        # torch.bernoulli expects a tensor of success probabilities
        keep_prob = 1.0 - self.p
        mask = torch.bernoulli(
            torch.full(x.shape, keep_prob, dtype=x.dtype, device=x.device)
        )

        # Apply mask and inverted scaling 
        # Divide by keep_prob so E[output] == E[input] regardless of p
        return x * mask / keep_prob
    
    def extra_repr(self) -> str:
        """Shows p value in print(model) output."""
        return f"p={self.p}"


# if __name__ == '__main__':
#     import math

#     print("=" * 55)
#     print("CustomDropout Unit Tests")
#     print("=" * 55)

#     torch.manual_seed(0)

#     # ── Test 1: Eval mode is identity ─────────────────────────────────────────
#     layer = CustomDropout(p=0.5)
#     layer.eval()
#     x = torch.randn(1000)
#     out = layer(x)
#     assert torch.equal(x, out), "FAIL: eval mode should be identity"
#     print("PASS  Test 1 — eval mode is identity")

#     # ── Test 2: Training mode zeros ~p fraction of elements ──────────────────
#     layer.train()
#     x = torch.ones(100_000)
#     out = layer(x)
#     zero_frac = (out == 0).float().mean().item()
#     assert math.isclose(zero_frac, 0.5, abs_tol=0.02), \
#         f"FAIL: expected ~50% zeros, got {zero_frac:.3f}"
#     print(f"PASS  Test 2 — ~{zero_frac*100:.1f}% zeros at p=0.5  (expected ~50%)")

#     # ── Test 3: Inverted scaling keeps expected value correct ─────────────────
#     x = torch.ones(100_000)
#     out = layer(x)
#     mean_val = out.mean().item()
#     assert math.isclose(mean_val, 1.0, abs_tol=0.02), \
#         f"FAIL: expected mean ~1.0 after scaling, got {mean_val:.4f}"
#     print(f"PASS  Test 3 — mean after scaling = {mean_val:.4f}  (expected ~1.0)")

#     # ── Test 4: Different p values ────────────────────────────────────────────
#     for p in (0.0, 0.2, 0.3, 0.8):
#         d = CustomDropout(p=p)
#         d.train()
#         x = torch.ones(100_000)
#         out = d(x)
#         zero_frac = (out == 0).float().mean().item()
#         mean_val  = out[out != 0].mean().item() if (out != 0).any() else 0.0
#         print(f"      p={p:.1f} → {zero_frac*100:.1f}% zeros | "
#               f"surviving mean = {mean_val:.4f}  (expected {1/(1-p) if p<1 else 'inf':.4f})")

#     # ── Test 5: Gradient flows through ───────────────────────────────────────
#     layer.train()
#     x = torch.randn(10, requires_grad=True)
#     out = layer(x)
#     loss = out.sum()
#     loss.backward()
#     assert x.grad is not None, "FAIL: no gradient on input"
#     print("PASS  Test 5 — gradients flow through correctly")

#     # ── Test 6: Invalid p raises error ───────────────────────────────────────
#     try:
#         CustomDropout(p=1.0)
#         print("FAIL  Test 6 — should have raised ValueError")
#     except ValueError:
#         print("PASS  Test 6 — p=1.0 raises ValueError correctly")

#     print("=" * 55)
#     print("All tests passed.")