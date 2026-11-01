"""
Modern PyTorch Training Pipeline for Gastrointestinal Polyp Segmentation.
Supports AdamW, Learning Rate Scheduling, Early Stopping, Checkpoint Persistence,
and Automated Metric History Tracking.
"""

import os
import time
import json
import csv
from typing import Dict, Any, Optional, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.models.registry import create_model
from ml.datasets.kvasir_dataset import (
    validate_kvasir_dataset,
    resolve_dataset_path,
    get_deterministic_splits,
    PolypSegmentationDataset
)
from ml.training.losses import DiceBCELoss
from ml.evaluation.metrics import calculate_segmentation_metrics, aggregate_dataset_metrics


class TrainingConfig:
    def __init__(
        self,
        model_name: str = "segformer",
        dataset_dir: Optional[str] = None,
        output_dir: str = "experiments/runs",
        image_size: int = 256,
        batch_size: int = 8,
        epochs: int = 20,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        random_seed: int = 42,
        device: str = "cpu",
        early_stopping_patience: int = 8
    ):
        self.model_name = model_name
        if not dataset_dir or dataset_dir == "auto":
            resolved, _ = resolve_dataset_path()
            self.dataset_dir = resolved
        else:
            self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_seed = random_seed
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.early_stopping_patience = early_stopping_patience

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "dataset_dir": self.dataset_dir,
            "output_dir": self.output_dir,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "random_seed": self.random_seed,
            "device": self.device,
            "early_stopping_patience": self.early_stopping_patience
        }


class PolypTrainer:
    def __init__(
        self,
        config: TrainingConfig,
        progress_callback: Optional[Any] = None,
        cancel_check: Optional[Any] = None,
        update_best_checkpoint: bool = True
    ):
        self.config = config
        self.progress_callback = progress_callback
        self.cancel_check = cancel_check
        self.update_best_checkpoint = update_best_checkpoint
        self._set_seed(config.random_seed)

        self.run_dir = os.path.join(
            config.output_dir,
            f"{config.model_name}_{int(time.time())}"
        )
        os.makedirs(self.run_dir, exist_ok=True)

        self.model = create_model(config.model_name, device=config.device, eval_mode=False)
        self.criterion = DiceBCELoss(bce_weight=0.5).to(config.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3
        )

        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
            "val_iou": [],
            "learning_rate": []
        }

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        count = 0

        for batch in dataloader:
            if self.cancel_check and self.cancel_check():
                break
            images = batch["image"].to(self.config.device)
            masks = batch["mask"].to(self.config.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            count += images.size(0)

        return total_loss / max(count, 1)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        metrics_list = []
        count = 0

        for batch in dataloader:
            images = batch["image"].to(self.config.device)
            masks = batch["mask"].to(self.config.device)

            logits = self.model(images)
            loss = self.criterion(logits, masks)
            total_loss += loss.item() * images.size(0)
            count += images.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            targets = masks.cpu().numpy()

            for i in range(images.size(0)):
                m = calculate_segmentation_metrics(targets[i, 0], probs[i, 0], threshold=0.5)
                metrics_list.append(m)

        agg = aggregate_dataset_metrics(metrics_list)
        return {
            "val_loss": total_loss / max(count, 1),
            "val_dice": agg.get("mean_dice", 0.0),
            "val_iou": agg.get("mean_iou", 0.0),
            "val_precision": agg.get("mean_precision", 0.0),
            "val_recall": agg.get("mean_recall", 0.0)
        }

    def run(self, max_batches: Optional[int] = None) -> Dict[str, Any]:
        """
        Executes full training pipeline or limited smoke test.
        """
        # Validate dataset using priority resolution
        dataset_dir = self.config.dataset_dir
        report = validate_kvasir_dataset(dataset_dir)
        if not report.is_valid:
            resolved_dir, resolved_report = resolve_dataset_path(dataset_dir)
            if resolved_report.is_valid:
                dataset_dir = resolved_dir
                report = resolved_report
            else:
                return {
                    "success": False,
                    "error": f"Invalid or missing dataset at '{self.config.dataset_dir}'. Found {len(report.valid_pairs)} valid pairs."
                }

        train_pairs, val_pairs, _ = get_deterministic_splits(
            report.valid_pairs, seed=self.config.random_seed
        )

        train_ds = PolypSegmentationDataset(
            train_pairs, image_size=(self.config.image_size, self.config.image_size), is_training=True
        )
        val_ds = PolypSegmentationDataset(
            val_pairs, image_size=(self.config.image_size, self.config.image_size), is_training=False
        )

        # Batch size clamped to available training pairs
        actual_batch_size = min(self.config.batch_size, max(len(train_pairs), 1))
        train_loader = DataLoader(train_ds, batch_size=actual_batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=actual_batch_size, shuffle=False)

        best_dice = -1.0
        best_iou = -1.0
        best_checkpoint_path = os.path.join(self.run_dir, "best_checkpoint.pt")
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            if self.cancel_check and self.cancel_check():
                break

            train_loss = self.train_epoch(train_loader)
            val_results = self.validate(val_loader)

            val_dice = val_results["val_dice"]
            val_loss = val_results["val_loss"]
            val_iou = val_results["val_iou"]
            lr = float(self.optimizer.param_groups[0]["lr"])

            self.scheduler.step(val_dice)

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(round(train_loss, 4))
            self.history["val_loss"].append(round(val_loss, 4))
            self.history["val_dice"].append(round(val_dice, 4))
            self.history["val_iou"].append(round(val_iou, 4))
            self.history["learning_rate"].append(lr)

            # Save best checkpoint
            is_best = val_dice > best_dice
            if is_best:
                best_dice = val_dice
                best_iou = val_iou
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "best_dice": best_dice,
                    "best_iou": val_iou,
                    "config": self.config.to_dict()
                }, best_checkpoint_path)

                # Safely update primary checkpoint in ml/checkpoints/ with .bak backup
                if self.update_best_checkpoint:
                    main_ckpt_dir = "ml/checkpoints"
                    os.makedirs(main_ckpt_dir, exist_ok=True)
                    main_ckpt_path = os.path.join(main_ckpt_dir, f"{self.config.model_name}_best.pt")
                    if os.path.exists(main_ckpt_path):
                        import shutil
                        bak_path = main_ckpt_path + ".bak"
                        if not os.path.exists(bak_path):
                            shutil.copy2(main_ckpt_path, bak_path)
                    torch.save({
                        "model_name": self.config.model_name,
                        "state_dict": self.model.state_dict(),
                        "epoch": epoch,
                        "best_dice": best_dice,
                        "best_iou": val_iou,
                        "trained_run": self.run_dir
                    }, main_ckpt_path)
            else:
                patience_counter += 1

            # Dispatch live progress update
            if self.progress_callback:
                self.progress_callback({
                    "epoch": epoch,
                    "total_epochs": self.config.epochs,
                    "train_loss": round(float(train_loss), 4),
                    "val_loss": round(float(val_loss), 4),
                    "val_dice": round(float(val_dice), 4),
                    "val_iou": round(float(val_iou), 4),
                    "learning_rate": lr,
                    "best_dice": round(float(best_dice), 4),
                    "progress_percent": round((epoch / self.config.epochs) * 100.0, 1),
                    "dataset_used": dataset_dir,
                    "dataset_pairs": len(report.valid_pairs)
                })

            if patience_counter >= self.config.early_stopping_patience:
                break

        # Save training history JSON
        history_path = os.path.join(self.run_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump({
                "config": self.config.to_dict(),
                "history": self.history,
                "best_dice": best_dice,
                "best_iou": best_iou,
                "best_checkpoint": best_checkpoint_path,
                "dataset_used": dataset_dir,
                "dataset_pairs": len(report.valid_pairs)
            }, f, indent=2)

        return {
            "success": True,
            "run_dir": self.run_dir,
            "best_dice": best_dice,
            "best_iou": best_iou,
            "best_checkpoint": best_checkpoint_path,
            "dataset_used": dataset_dir,
            "dataset_pairs": len(report.valid_pairs),
            "history": self.history
        }
