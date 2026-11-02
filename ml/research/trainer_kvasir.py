"""
Dedicated, Reproducible Kvasir-SEG Full Dataset Trainer for SegFormer Research Baseline.
Trains on 700 actual Kvasir-SEG training pairs, validates on 200 validation pairs.
Saves new checkpoint separately to: ml/checkpoints/segformer_kvasir_full_research.pt
Never touches or overwrites existing baseline checkpoints.
"""

import os
import sys
import time
import json
import random
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from ml.datasets.kvasir_dataset import (
    PolypSegmentationDataset,
    resolve_dataset_path,
    get_deterministic_splits
)
from ml.models.segformer import SegFormer
from ml.training.losses import DiceBCELoss
from ml.evaluation.metrics import calculate_segmentation_metrics


def train_kvasir_research_model(
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 0.0003,
    device: str = "cpu",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Executes reproducible training on the genuine Kvasir-SEG dataset.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"\n=======================================================")
    print(f"  KVASIR-SEG RESEARCH TRAINING (SegFormer-B0)")
    print(f"  Hardware: {device.upper()} | Seed: {seed} | Epochs: {epochs}")
    print(f"=======================================================\n")

    # 1. Resolve Dataset
    dataset_path, report = resolve_dataset_path()
    if not report.is_valid or len(report.valid_pairs) < 100:
        raise RuntimeError(f"Kvasir-SEG dataset invalid or missing at: {dataset_path}")

    train_pairs, val_pairs, test_pairs = get_deterministic_splits(report.valid_pairs, seed=seed)
    print(f"Resolved Dataset: {dataset_path}")
    print(f"Total: {len(report.valid_pairs)} | Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")

    # 2. Dataloaders
    train_dataset = PolypSegmentationDataset(
        pairs=train_pairs,
        image_size=(256, 256),
        is_training=True,
        normalize_imagenet=True
    )
    val_dataset = PolypSegmentationDataset(
        pairs=val_pairs,
        image_size=(256, 256),
        is_training=False,
        normalize_imagenet=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Model & Optimizer
    model = SegFormer(in_channels=3, num_classes=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = DiceBCELoss(bce_weight=0.5)

    history = []
    best_val_dice = -1.0
    start_total_time = time.perf_counter()

    os.makedirs("ml/checkpoints", exist_ok=True)
    save_path = "ml/checkpoints/segformer_kvasir_full_research.pt"

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  Epoch [{epoch}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        train_loss = running_loss / max(num_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_dice_list = []
        val_iou_list = []
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item()
                val_batches += 1

                probs = torch.sigmoid(logits).cpu().numpy()
                gt_np = masks.cpu().numpy()

                for b in range(probs.shape[0]):
                    p_bin = (probs[b, 0] >= 0.5).astype(np.uint8)
                    g_bin = (gt_np[b, 0] >= 0.5).astype(np.uint8)
                    m = calculate_segmentation_metrics(g_bin, p_bin)
                    val_dice_list.append(m["dice"])
                    val_iou_list.append(m["iou"])

        mean_val_loss = val_loss / max(val_batches, 1)
        mean_val_dice = float(np.mean(val_dice_list))
        mean_val_iou = float(np.mean(val_iou_list))
        epoch_duration = round(time.perf_counter() - epoch_start, 2)

        print(f"--> Epoch {epoch}/{epochs} Complete ({epoch_duration}s) | Train Loss: {train_loss:.4f} | Val Loss: {mean_val_loss:.4f} | Val Dice: {mean_val_dice:.4f} | Val IoU: {mean_val_iou:.4f}\n")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(mean_val_loss, 4),
            "val_dice": round(mean_val_dice, 4),
            "val_iou": round(mean_val_iou, 4),
            "epoch_duration_seconds": epoch_duration
        })

        if mean_val_dice > best_val_dice:
            best_val_dice = mean_val_dice
            torch.save({
                "model_name": "segformer_kvasir_full_research",
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val_dice": mean_val_dice,
                "best_val_iou": mean_val_iou,
                "training_dataset": "Kvasir-SEG (700 train pairs)",
                "sample_trained": False,
                "created_at": time.time(),
                "device": device
            }, save_path)
            print(f"  [SAVED CHECKPOINT] -> {save_path} (Val Dice: {mean_val_dice:.4f})")

    total_duration = round(time.perf_counter() - start_total_time, 2)
    os.makedirs("experiments", exist_ok=True)
    log_path = "experiments/kvasir_full_training_log.json"

    summary = {
        "model": "SegFormer-B0",
        "dataset": "Kvasir-SEG (1,000 images)",
        "train_samples": len(train_pairs),
        "val_samples": len(val_pairs),
        "test_samples": len(test_pairs),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": device,
        "seed": seed,
        "total_training_seconds": total_duration,
        "best_val_dice": round(best_val_dice, 4),
        "checkpoint_path": save_path,
        "history": history
    }

    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining Log saved to: {log_path}")
    print(f"Total training time: {total_duration}s ({total_duration/60:.2f} mins)")
    return summary


if __name__ == "__main__":
    train_kvasir_research_model(epochs=2, batch_size=4, device="cpu", seed=42)
