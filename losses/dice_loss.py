"""
Dice Loss for Segmentation
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for multi-class segmentation.
    Loss = 1 - (2 * |X ∩ Y| + eps) / (|X| + |Y| + eps)

    Justification:
        Dice Loss directly optimises the overlap metric and is robust
        to class imbalance - critical for trimaps where background
        pixels dominate. Combined with CrossEntropy for stable gradients.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,   # (B, C, H, W) raw logits
        targets: torch.Tensor,  # (B, H, W)    integer class labels
    ) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)       # (B, C, H, W)

        # One-hot encode targets → (B, C, H, W)
        targets_oh  = torch.zeros_like(probs)
        targets_oh.scatter_(1, targets.unsqueeze(1), 1.0)

        # Dice per class, averaged
        dims  = (0, 2, 3)   # sum over batch and spatial dims
        inter = (probs * targets_oh).sum(dim=dims)           # (C,)
        union = (probs + targets_oh).sum(dim=dims)           # (C,)
        dice  = (2.0 * inter + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()