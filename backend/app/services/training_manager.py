"""
Background Training Manager for Asynchronous PyTorch Model Optimization.
Provides thread-safe state tracking, live metric callbacks, and graceful cancellation.
"""

import os
import time
import threading
from typing import Dict, Any, Optional, List

from ml.training.trainer import TrainingConfig, PolypTrainer
from ml.datasets.kvasir_dataset import resolve_dataset_path


class TrainingManager:
    """
    Singleton manager running PyTorch training in a dedicated background daemon thread.
    """
    _instance: Optional["TrainingManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_state()
            return cls._instance

    def _init_state(self):
        self.state_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.status: str = "idle"  # "idle", "running", "completed", "stopped", "failed"
        self.model_name: Optional[str] = None
        self.current_epoch: int = 0
        self.total_epochs: int = 0
        self.progress_percent: float = 0.0
        self.train_loss: Optional[float] = None
        self.val_loss: Optional[float] = None
        self.val_dice: Optional[float] = None
        self.val_iou: Optional[float] = None
        self.learning_rate: Optional[float] = None
        self.best_dice: Optional[float] = None
        self.start_time: float = 0.0
        self.elapsed_time_sec: float = 0.0
        self.history: List[Dict[str, Any]] = []
        self.error_message: Optional[str] = None
        self.checkpoint_path: Optional[str] = None
        self.dataset_used: Optional[str] = None
        self.dataset_pairs: int = 0

    def start_training(
        self,
        model_name: str = "segformer",
        epochs: int = 10,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        dataset_source: str = "auto",
        early_stopping_patience: int = 5,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        with self.state_lock:
            if self.status == "running":
                return {
                    "success": False,
                    "message": f"Training already in progress for '{self.model_name}' (epoch {self.current_epoch}/{self.total_epochs})."
                }

            self._stop_event.clear()
            self.status = "running"
            self.model_name = model_name
            self.current_epoch = 0
            self.total_epochs = epochs
            self.progress_percent = 0.0
            self.train_loss = None
            self.val_loss = None
            self.val_dice = None
            self.val_iou = None
            self.learning_rate = learning_rate
            self.best_dice = None
            self.start_time = time.time()
            self.elapsed_time_sec = 0.0
            self.history = []
            self.error_message = None
            self.checkpoint_path = None
            self.dataset_used = None
            self.dataset_pairs = 0

        # Resolve dataset directory using strict priority
        if dataset_source == "sample_images":
            dataset_dir, report = resolve_dataset_path(preferred_path="data/sample_images")
        elif dataset_source in ("auto", "kvasir_seg", None, ""):
            dataset_dir, report = resolve_dataset_path()
        else:
            dataset_dir, report = resolve_dataset_path(preferred_path=dataset_source)

        config = TrainingConfig(
            model_name=model_name,
            dataset_dir=dataset_dir,
            output_dir="experiments/runs",
            image_size=256,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            early_stopping_patience=early_stopping_patience
        )

        self._thread = threading.Thread(
            target=self._run_training_worker,
            args=(config,),
            daemon=True
        )
        self._thread.start()

        return {
            "success": True,
            "message": f"Started training {model_name} for {epochs} epochs on {dataset_dir}.",
            "model_name": model_name,
            "epochs": epochs
        }

    def _run_training_worker(self, config: TrainingConfig):
        def on_progress(p: Dict[str, Any]):
            with self.state_lock:
                self.current_epoch = p["epoch"]
                self.total_epochs = p["total_epochs"]
                self.train_loss = p["train_loss"]
                self.val_loss = p["val_loss"]
                self.val_dice = p["val_dice"]
                self.val_iou = p["val_iou"]
                self.learning_rate = p["learning_rate"]
                self.best_dice = p["best_dice"]
                self.progress_percent = p["progress_percent"]
                self.dataset_used = p.get("dataset_used")
                self.dataset_pairs = p.get("dataset_pairs", 0)
                self.elapsed_time_sec = round(time.time() - self.start_time, 1)

                self.history.append({
                    "epoch": p["epoch"],
                    "train_loss": p["train_loss"],
                    "val_loss": p["val_loss"],
                    "val_dice": p["val_dice"],
                    "val_iou": p["val_iou"],
                    "learning_rate": p["learning_rate"]
                })

        def is_cancelled() -> bool:
            return self._stop_event.is_set()

        try:
            trainer = PolypTrainer(
                config=config,
                progress_callback=on_progress,
                cancel_check=is_cancelled,
                update_best_checkpoint=True
            )
            result = trainer.run()

            with self.state_lock:
                self.elapsed_time_sec = round(time.time() - self.start_time, 1)
                if is_cancelled():
                    self.status = "stopped"
                elif not result.get("success", False):
                    self.status = "failed"
                    self.error_message = result.get("error", "Unknown training error")
                else:
                    self.status = "completed"
                    self.progress_percent = 100.0
                    self.checkpoint_path = result.get("best_checkpoint")
                    self.best_dice = result.get("best_dice")

        except Exception as e:
            with self.state_lock:
                self.status = "failed"
                self.error_message = str(e)
                self.elapsed_time_sec = round(time.time() - self.start_time, 1)

    def stop_training(self) -> Dict[str, Any]:
        with self.state_lock:
            if self.status != "running":
                return {"success": False, "message": "No training job currently running."}
            self._stop_event.set()
            return {"success": True, "message": "Cancellation requested. Halting after current batch/epoch."}

    def get_status(self) -> Dict[str, Any]:
        with self.state_lock:
            elapsed = self.elapsed_time_sec
            if self.status == "running":
                elapsed = round(time.time() - self.start_time, 1)

            return {
                "status": self.status,
                "model_name": self.model_name,
                "current_epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "progress_percent": self.progress_percent,
                "train_loss": self.train_loss,
                "val_loss": self.val_loss,
                "val_dice": self.val_dice,
                "val_iou": self.val_iou,
                "learning_rate": self.learning_rate,
                "best_dice": self.best_dice,
                "elapsed_time_sec": elapsed,
                "history": list(self.history),
                "error_message": self.error_message,
                "checkpoint_path": self.checkpoint_path,
                "dataset_used": self.dataset_used,
                "dataset_pairs": self.dataset_pairs
            }


# Singleton accessor
training_manager = TrainingManager()
