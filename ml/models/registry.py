"""
Unified Model Registry and Factory for Polyp Segmentation.
Supports dynamic loading, parameter inspection, checkpoint handling, and device placement.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from ml.models.unet import UNet
from ml.models.linknet import LinkNet
from ml.models.segformer import SegFormer


MODEL_CATALOG = {
    "segformer": {
        "id": "segformer",
        "name": "SegFormer-B0",
        "family": "Transformer",
        "description": "Hierarchical Mix Transformer (MiT) encoder with an All-MLP decoder for polyp boundary delineation.",
        "input_resolution": "256x256",
        "is_modern": True,
        "historical_reference": None,
        "class": SegFormer
    },
    "unet": {
        "id": "unet",
        "name": "U-Net Baseline",
        "family": "CNN (Encoder-Decoder)",
        "description": "Biomedical U-Net baseline featuring symmetric contracting/expanding paths with concatenative skip connections.",
        "input_resolution": "256x256",
        "is_modern": False,
        "historical_reference": "Report Table 5.4.1 (EfficientNetB7: Dice 0.8882, IoU 0.8118; ResNet34: Dice 0.8253, IoU 0.7341)",
        "class": UNet
    },
    "linknet": {
        "id": "linknet",
        "name": "LinkNet Baseline",
        "family": "CNN (Additive Bypass)",
        "description": "Lightweight semantic segmentation architecture using additive feature fusion for rapid clinical turnaround.",
        "input_resolution": "256x256",
        "is_modern": False,
        "historical_reference": "Report Table 5.4.2 (EfficientNetB7: Dice 0.8742, IoU 0.8007; ResNet34: Dice 0.7953, IoU 0.6941)",
        "class": LinkNet
    }
}


def count_parameters(model: nn.Module) -> int:
    """Returns total number of trainable and non-trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: nn.Module) -> float:
    """Returns approximate model weight size in megabytes (FP32)."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def create_model(
    model_name: str,
    device: str = "cpu",
    checkpoint_path: Optional[str] = None,
    eval_mode: bool = True
) -> nn.Module:
    """
    Instantiates requested model, places on target device, and optionally loads checkpoint.
    """
    model_key = model_name.lower().strip()
    if model_key not in MODEL_CATALOG:
        valid_keys = list(MODEL_CATALOG.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available architectures: {valid_keys}")

    model_cls = MODEL_CATALOG[model_key]["class"]
    model = model_cls(in_channels=3, num_classes=1)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)

    model = model.to(device)
    if eval_mode:
        model.eval()

    return model


def get_available_models_metadata() -> List[Dict[str, Any]]:
    """Returns metadata for all supported models including parameter counts."""
    metadata_list = []
    for key, info in MODEL_CATALOG.items():
        # Instantiate dummy instance to compute exact parameters
        temp_model = info["class"](in_channels=3, num_classes=1)
        params = count_parameters(temp_model)
        size_mb = get_model_size_mb(temp_model)
        del temp_model

        metadata_list.append({
            "id": info["id"],
            "name": info["name"],
            "family": info["family"],
            "description": info["description"],
            "input_resolution": info["input_resolution"],
            "parameters": params,
            "parameters_formatted": f"{params / 1e6:.2f}M",
            "size_mb": round(size_mb, 2),
            "is_modern": info["is_modern"],
            "historical_reference": info["historical_reference"]
        })
    return metadata_list
