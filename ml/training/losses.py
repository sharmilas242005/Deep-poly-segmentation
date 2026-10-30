"""
Loss Functions for Medical Image Polyp Segmentation.

Implements combined Dice + Binary Cross-Entropy (DiceBCELoss).
Polyps occupy only a small fraction of endoscopic frames (class imbalance).
Standard BCE penalizes background errors evenly, while Dice loss directly
maximizes foreground region overlap.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss computed on sigmoid probabilities.
    Dice = (2 * intersection + eps) / (sum_true + sum_pred + eps)
    Loss = 1 - Dice
    """
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions: (B, C, H, W) -> (B, -1)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = torch.sum(probs_flat * targets_flat, dim=1)
        cardinality = torch.sum(probs_flat + targets_flat, dim=1)

        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return torch.mean(1.0 - dice)


class DiceBCELoss(nn.Module):
    """
    Hybrid Dice + BCE Loss:
    L = alpha * BCEWithLogits + (1 - alpha) * DiceLoss
    """
    def __init__(self, bce_weight: float = 0.5, eps: float = 1e-7):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(eps=eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss
