"""
Unified Model Registry and Factory for Polyp Segmentation.
Supports dynamic loading, parameter inspection, checkpoint handling, device placement,
and robust verification of trained model weights.
"""

import os
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn

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


def get_checkpoint_info(model_name: str, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Inspects checkpoint existence, file size, parameter counts, and training distribution.
    Identifies whether weights are trained on the 3 demo cases or missing.
    """
    key = model_name.lower().strip()
    if key not in MODEL_CATALOG:
        raise ValueError(f"Unknown model '{model_name}'. Available architectures: {list(MODEL_CATALOG.keys())}")

    resolved_path = checkpoint_path
    if resolved_path is None:
        default_ckpt = os.path.join("ml", "checkpoints", f"{key}_best.pt")
        if os.path.isfile(default_ckpt):
            resolved_path = default_ckpt

    if resolved_path and os.path.isfile(resolved_path):
        size_mb = round(os.path.getsize(resolved_path) / (1024 * 1024), 2)
        filename = os.path.basename(resolved_path)
        return {
            "checkpoint_available": True,
            "checkpoint_filename": filename,
            "checkpoint_path": os.path.abspath(resolved_path),
            "is_trained": True,
            "training_type": "sample_trained",
            "training_dataset": "3 Demo Cases (data/sample_images)",
            "dataset_details": "Trained on 3 sample endoscopy images in data/sample_images. NOT trained on the full 1,000-image Kvasir-SEG dataset.",
            "status_message": "Sample-trained weights loaded (3 demo cases)",
            "checkpoint_size_mb": size_mb
        }

    return {
        "checkpoint_available": False,
        "checkpoint_filename": None,
        "checkpoint_path": None,
        "is_trained": False,
        "training_type": "missing",
        "training_dataset": None,
        "dataset_details": None,
        "status_message": "Trained checkpoint not found. Please provide trained model weights before running inference.",
        "checkpoint_size_mb": 0.0
    }


def load_checkpoint_weights(
    model: nn.Module,
    checkpoint_path: str,
    device: str = "cpu"
) -> nn.Module:
    """
    Safely unpackages and strictly validates model weights from checkpoint file.
    Supports formats:
      - Dictionary containing 'state_dict'
      - Dictionary containing 'model_state_dict'
      - Raw state dictionary of parameter tensors
    Raises FileNotFoundError if checkpoint missing, or RuntimeError if weights incompatible.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Trained checkpoint not found at '{checkpoint_path}'. "
            "Please provide trained model weights before running inference."
        )

    try:
        raw_checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to read checkpoint file '{checkpoint_path}': {str(e)}")

    if isinstance(raw_checkpoint, dict):
        if "state_dict" in raw_checkpoint and isinstance(raw_checkpoint["state_dict"], dict):
            sd = raw_checkpoint["state_dict"]
        elif "model_state_dict" in raw_checkpoint and isinstance(raw_checkpoint["model_state_dict"], dict):
            sd = raw_checkpoint["model_state_dict"]
        else:
            sd = raw_checkpoint
    else:
        raise ValueError(
            f"Invalid checkpoint content in '{checkpoint_path}'. Expected dictionary, got {type(raw_checkpoint)}"
        )

    # Strip DataParallel 'module.' prefix if present
    cleaned_sd = {}
    for k, v in sd.items():
        clean_key = k[7:] if k.startswith("module.") else k
        cleaned_sd[clean_key] = v

    try:
        model.load_state_dict(cleaned_sd, strict=True)
    except Exception as e:
        raise RuntimeError(
            f"Checkpoint weights in '{checkpoint_path}' are incompatible with model architecture: {str(e)}"
        )

    model._is_trained = True
    model._checkpoint_path = checkpoint_path
    return model


def create_model(
    model_name: str,
    device: str = "cpu",
    checkpoint_path: Optional[str] = None,
    eval_mode: bool = True,
    require_checkpoint: bool = False
) -> nn.Module:
    """
    Instantiates requested model, places on target device, and optionally loads checkpoint.
    If require_checkpoint is True, raises FileNotFoundError if weights cannot be loaded,
    preventing any silent fallback to uninitialized random weights.
    """
    model_key = model_name.lower().strip()
    if model_key not in MODEL_CATALOG:
        valid_keys = list(MODEL_CATALOG.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available architectures: {valid_keys}")

    model_cls = MODEL_CATALOG[model_key]["class"]
    model = model_cls(in_channels=3, num_classes=1)

    if require_checkpoint:
        if checkpoint_path is None:
            default_ckpt = os.path.join("ml", "checkpoints", f"{model_key}_best.pt")
            if os.path.isfile(default_ckpt):
                checkpoint_path = default_ckpt
            else:
                raise FileNotFoundError(
                    "Trained checkpoint not found. Please provide trained model weights before running inference."
                )
        load_checkpoint_weights(model, checkpoint_path, device=device)
    elif checkpoint_path is not None:
        load_checkpoint_weights(model, checkpoint_path, device=device)
    else:
        # Created without checkpoint (e.g. for parameter calculation or training initialization)
        model._is_trained = False
        model._checkpoint_path = None

    model = model.to(device)
    if eval_mode:
        model.eval()

    return model


def get_available_models_metadata() -> List[Dict[str, Any]]:
    """Returns metadata for all supported models including parameters, size, and checkpoint status."""
    metadata_list = []
    for key, info in MODEL_CATALOG.items():
        # Instantiate temporary instance to compute exact architectural parameters
        temp_model = info["class"](in_channels=3, num_classes=1)
        params = count_parameters(temp_model)
        size_mb = get_model_size_mb(temp_model)
        del temp_model

        ckpt_info = get_checkpoint_info(key)

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
            "historical_reference": info["historical_reference"],
            "checkpoint_available": ckpt_info["checkpoint_available"],
            "checkpoint_filename": ckpt_info["checkpoint_filename"],
            "checkpoint_path": ckpt_info["checkpoint_path"],
            "is_trained": ckpt_info["is_trained"],
            "training_type": ckpt_info["training_type"],
            "training_dataset": ckpt_info["training_dataset"],
            "dataset_details": ckpt_info["dataset_details"],
            "status_message": ckpt_info["status_message"],
            "checkpoint_size_mb": ckpt_info["checkpoint_size_mb"]
        })
    return metadata_list
