"""Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.

    Loss = 1 - IoU  (per sample)

    Steps:
        1. Convert (cx, cy, w, h) to (x1, y1, x2, y2) corner format
        2. Compute intersection area
        3. Compute union area = area_pred + area_target - intersection
        4. IoU = intersection / (union + eps)
        5. Loss = 1 - IoU

    Args:
        eps       : small constant for numerical stability 
        reduction : 'mean' | 'sum' | 'none'  
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()

         # Validate reduction
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                f"reduction must be 'none', 'mean', or 'sum', got '{reduction}'"
            )
        
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format.
        """
        # Step 1: cx,cy,w,h to x1,y1,x2,y2 
        # pred
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2   # cx - w/2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2   # cy - h/2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2   # cx + w/2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2   # cy + h/2

        # target
        tgt_x1  = target_boxes[:, 0] - target_boxes[:, 2] / 2
        tgt_y1  = target_boxes[:, 1] - target_boxes[:, 3] / 2
        tgt_x2  = target_boxes[:, 0] + target_boxes[:, 2] / 2
        tgt_y2  = target_boxes[:, 1] + target_boxes[:, 3] / 2

        # Step 2: Intersection 
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)

        # clamp to 0 - no negative side lengths if boxes don't overlap
        inter_w    = (inter_x2 - inter_x1).clamp(min=0)
        inter_h    = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h  # (B,)

        # Step 3: Union 
        pred_area  = (pred_x2 - pred_x1) * (pred_y2 - pred_y1) # (B,)
        tgt_area   = (tgt_x2 - tgt_x1) * (tgt_y2 - tgt_y1)  # (B,)
        union_area = pred_area + tgt_area - inter_area  # (B,)

        # Step 4: IoU 
        iou  = inter_area / (union_area + self.eps)  # (B,)

        # Step 5: Loss = 1 - IoU 
        loss = 1.0 - iou  # (B,)

        # Reduction 
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:                          # 'none'
            return loss
        

# Unit tests 

# if __name__ == '__main__':
#     import math

#     print("=" * 55)
#     print("IoULoss Unit Tests")
#     print("=" * 55)

#     criterion = IoULoss(eps=1e-6, reduction='mean')

#     # ── Test 1: Perfect overlap → loss = 0 ───────────────────────────────────
#     boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
#     loss  = criterion(boxes, boxes)
#     assert math.isclose(loss.item(), 0.0, abs_tol=1e-5), \
#         f"FAIL: perfect overlap should give loss=0, got {loss.item()}"
#     print(f"PASS  Test 1 — perfect overlap → loss = {loss.item():.6f}")

#     # ── Test 2: No overlap → loss = 1 ────────────────────────────────────────
#     pred   = torch.tensor([[0.1, 0.1, 0.1, 0.1]])   # top-left box
#     target = torch.tensor([[0.9, 0.9, 0.1, 0.1]])   # bottom-right box
#     loss   = criterion(pred, target)
#     assert math.isclose(loss.item(), 1.0, abs_tol=1e-5), \
#         f"FAIL: no overlap should give loss=1, got {loss.item()}"
#     print(f"PASS  Test 2 — no overlap      → loss = {loss.item():.6f}")

#     # ── Test 3: Partial overlap → loss in (0, 1) ─────────────────────────────
#     pred   = torch.tensor([[0.4, 0.5, 0.4, 0.6]])
#     target = torch.tensor([[0.6, 0.5, 0.4, 0.6]])
#     loss   = criterion(pred, target)
#     assert 0.0 < loss.item() < 1.0, \
#         f"FAIL: partial overlap loss should be in (0,1), got {loss.item()}"
#     print(f"PASS  Test 3 — partial overlap → loss = {loss.item():.6f}")

#     # ── Test 4: Gradients flow ────────────────────────────────────────────────
#     pred   = torch.tensor([[0.5, 0.5, 0.3, 0.3]], requires_grad=True)
#     target = torch.tensor([[0.6, 0.6, 0.3, 0.3]])
#     loss   = criterion(pred, target)
#     loss.backward()
#     assert pred.grad is not None, "FAIL: no gradient on pred_boxes"
#     print(f"PASS  Test 4 — gradients flow  → grad = {pred.grad.numpy()}")

#     # ── Test 5: Batch of boxes ────────────────────────────────────────────────
#     pred   = torch.rand(8, 4).clamp(0.1, 0.9)
#     target = torch.rand(8, 4).clamp(0.1, 0.9)
#     loss   = criterion(pred, target)
#     assert loss.shape == torch.Size([]), \
#         f"FAIL: mean reduction should return scalar, got {loss.shape}"
#     print(f"PASS  Test 5 — batch of 8      → loss = {loss.item():.6f}")

#     # ── Test 6: reduction='none' returns (B,) ─────────────────────────────────
#     criterion_none = IoULoss(reduction='none')
#     loss = criterion_none(pred, target)
#     assert loss.shape == torch.Size([8]), \
#         f"FAIL: 'none' reduction should return (B,), got {loss.shape}"
#     print(f"PASS  Test 6 — reduction=none  → shape = {tuple(loss.shape)}")

#     # ── Test 7: Invalid reduction raises error ────────────────────────────────
#     try:
#         IoULoss(reduction='invalid')
#         print("FAIL  Test 7 — should have raised ValueError")
#     except ValueError:
#         print("PASS  Test 7 — invalid reduction raises ValueError")

#     print("=" * 55)
#     print("All tests passed.")
        
