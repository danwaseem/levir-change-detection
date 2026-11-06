from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    numerator = 2.0 * (prob * target).sum(dims) + eps
    denominator = prob.sum(dims) + target.sum(dims) + eps
    return 1.0 - (numerator / denominator).mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, kind: str = "bce_dice") -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    if kind == "bce":
        return bce
    if kind == "bce_dice":
        return bce + soft_dice_loss(logits, target)
    raise ValueError(f"Unknown loss kind: {kind}")
