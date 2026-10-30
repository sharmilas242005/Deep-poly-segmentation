"""
Fine-tunes lightweight initial checkpoints on endoscopic polyp sample frames
so that demonstrable checkpoints exist in ml/checkpoints/.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
from PIL import Image
import numpy as np

from ml.models.registry import create_model, count_parameters
from ml.datasets.kvasir_dataset import PolypSegmentationDataset
from ml.training.losses import DiceBCELoss


def train_quick_checkpoints():
    os.makedirs("ml/checkpoints", exist_ok=True)
    device = "cpu"

    sample_dir = "data/sample_images"
    samples = [
        ("sample_polyp_sessile.jpg", "sample_polyp_sessile_mask.png"),
        ("sample_polyp_elevated.jpg", "sample_polyp_elevated_mask.png"),
        ("sample_normal_mucosa.jpg", "sample_normal_mucosa_mask.png"),
    ]
    pairs = [
        (os.path.join(sample_dir, img), os.path.join(sample_dir, msk))
        for img, msk in samples
        if os.path.exists(os.path.join(sample_dir, img)) and os.path.exists(os.path.join(sample_dir, msk))
    ]

    if not pairs:
        print("No sample pairs found.")
        return

    dataset = PolypSegmentationDataset(pairs, image_size=(256, 256), is_training=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    criterion = DiceBCELoss(bce_weight=0.5)

    for model_name in ["segformer", "unet", "linknet"]:
        print(f"Training demo checkpoint for {model_name}...")
        model = create_model(model_name, device=device, eval_mode=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        for epoch in range(12):
            model.train()
            for batch in loader:
                imgs = batch["image"].to(device)
                msks = batch["mask"].to(device)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, msks)
                loss.backward()
                optimizer.step()

        ckpt_path = os.path.join("ml/checkpoints", f"{model_name}_best.pt")
        torch.save({
            "model_name": model_name,
            "state_dict": model.state_dict(),
            "sample_trained": True,
            "parameters": count_parameters(model)
        }, ckpt_path)
        print(f"Saved {model_name} checkpoint to {ckpt_path}")


if __name__ == "__main__":
    train_quick_checkpoints()
