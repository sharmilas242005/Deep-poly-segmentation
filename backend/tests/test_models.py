"""
Unit tests for PyTorch models: U-Net, LinkNet, SegFormer, and Model Registry.
"""

import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from ml.models.unet import UNet
from ml.models.linknet import LinkNet
from ml.models.segformer import SegFormer
from ml.models.registry import create_model, get_available_models_metadata, count_parameters


def test_unet_forward():
    model = UNet(in_channels=3, num_classes=1)
    dummy_input = torch.randn(2, 3, 256, 256)
    out = model(dummy_input)
    assert out.shape == (2, 1, 256, 256), f"Expected (2, 1, 256, 256), got {out.shape}"

    mask = model.predict_mask(dummy_input)
    assert mask.shape == (2, 1, 256, 256)
    assert torch.all((mask == 0.0) | (mask == 1.0))


def test_linknet_forward():
    model = LinkNet(in_channels=3, num_classes=1)
    dummy_input = torch.randn(2, 3, 256, 256)
    out = model(dummy_input)
    assert out.shape == (2, 1, 256, 256), f"Expected (2, 1, 256, 256), got {out.shape}"

    mask = model.predict_mask(dummy_input)
    assert mask.shape == (2, 1, 256, 256)
    assert torch.all((mask == 0.0) | (mask == 1.0))


def test_segformer_forward():
    model = SegFormer(in_channels=3, num_classes=1)
    dummy_input = torch.randn(2, 3, 256, 256)
    out = model(dummy_input)
    assert out.shape == (2, 1, 256, 256), f"Expected (2, 1, 256, 256), got {out.shape}"

    mask = model.predict_mask(dummy_input)
    assert mask.shape == (2, 1, 256, 256)
    assert torch.all((mask == 0.0) | (mask == 1.0))


def test_registry_metadata():
    metadata = get_available_models_metadata()
    assert len(metadata) == 3
    ids = [m["id"] for m in metadata]
    assert "segformer" in ids
    assert "unet" in ids
    assert "linknet" in ids

    for m in metadata:
        assert m["parameters"] > 100_000
        assert m["size_mb"] > 0.5


def test_registry_factory():
    for name in ["unet", "linknet", "segformer"]:
        model = create_model(name, device="cpu")
        assert isinstance(model, torch.nn.Module)
        p = count_parameters(model)
        assert p > 0

    with pytest.raises(ValueError):
        create_model("non_existent_model")
