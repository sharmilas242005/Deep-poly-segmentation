"""
Unit and integration tests for Preprocessing, Checkpoint Loading, and Edge-Case API logic.
Part of Phase 6 and Phase 7 verification.
"""

import os
import sys
import tempfile
import torch
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from ml.datasets.kvasir_dataset import (
    PolypSegmentationDataset,
    validate_kvasir_dataset,
    get_deterministic_splits
)
from ml.models.registry import (
    create_model,
    count_parameters,
    get_model_size_mb,
    MODEL_CATALOG
)
from backend.app.main import app

client = TestClient(app)


# =========================================================================
# 1. PREPROCESSING TESTS
# =========================================================================

def test_nearest_neighbor_mask_interpolation():
    """
    Verifies that ground-truth mask resizing strictly uses nearest-neighbor interpolation,
    ensuring no intermediate fractional/fuzzy boundary pixels are introduced.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a tiny 4x4 binary mask with a 2x2 white square in the center
        mask_arr = np.zeros((4, 4), dtype=np.uint8)
        mask_arr[1:3, 1:3] = 255
        mask_pil = Image.fromarray(mask_arr, mode="L")
        mask_path = os.path.join(tmpdir, "test_mask.png")
        mask_pil.save(mask_path)

        # Create dummy RGB image
        img_arr = np.full((4, 4, 3), 120, dtype=np.uint8)
        img_pil = Image.fromarray(img_arr, mode="RGB")
        img_path = os.path.join(tmpdir, "test_img.jpg")
        img_pil.save(img_path)

        dataset = PolypSegmentationDataset(
            pairs=[(img_path, mask_path)],
            image_size=(256, 256),
            is_training=False
        )

        sample = dataset[0]
        mask_tensor = sample["mask"]

        # The mask must only contain binary values 0.0 or 1.0, nothing in between
        unique_vals = torch.unique(mask_tensor).tolist()
        for v in unique_vals:
            assert v in [0.0, 1.0], f"Found non-binary value {v} in resized mask!"


def test_imagenet_normalization_applied():
    """
    Verifies that input image tensors are normalized according to ImageNet statistics.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Solid white image (255, 255, 255)
        img_arr = np.full((64, 64, 3), 255, dtype=np.uint8)
        img_pil = Image.fromarray(img_arr, mode="RGB")
        img_path = os.path.join(tmpdir, "test_white.jpg")
        img_pil.save(img_path)

        mask_arr = np.zeros((64, 64), dtype=np.uint8)
        mask_path = os.path.join(tmpdir, "test_mask.png")
        Image.fromarray(mask_arr, mode="L").save(mask_path)

        dataset = PolypSegmentationDataset(
            pairs=[(img_path, mask_path)],
            image_size=(128, 128),
            is_training=False,
            normalize_imagenet=True
        )

        sample = dataset[0]
        img_tensor = sample["image"]

        # ImageNet mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
        # (1.0 - 0.485) / 0.229 ~= 2.2489
        expected_r = (1.0 - 0.485) / 0.229
        assert abs(img_tensor[0, 0, 0].item() - expected_r) < 0.05


def test_deterministic_split_reproducibility():
    """
    Verifies that get_deterministic_splits produces identical train/val/test
    partitions given the same seed.
    """
    dummy_pairs = [(f"img_{i}.jpg", f"mask_{i}.png") for i in range(20)]

    train1, val1, test1 = get_deterministic_splits(dummy_pairs, seed=42)
    train2, val2, test2 = get_deterministic_splits(dummy_pairs, seed=42)

    assert train1 == train2
    assert val1 == val2
    assert test1 == test2

    # Verify no overlap between splits
    train_set = set(train1)
    val_set = set(val1)
    test_set = set(test1)

    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0
    assert len(train_set) + len(val_set) + len(test_set) == len(dummy_pairs)


def test_validate_kvasir_dataset_samples():
    """
    Verifies that validate_kvasir_dataset accurately identifies the sample_images dataset.
    """
    report = validate_kvasir_dataset("data/sample_images")
    assert report.is_valid is True
    assert len(report.valid_pairs) == 3
    assert len(report.corrupted_files) == 0
    assert len(report.dimension_mismatches) == 0


def test_validate_kvasir_dataset_nonexistent():
    """
    Verifies graceful empty report when dataset path does not exist.
    """
    report = validate_kvasir_dataset("data/path_that_does_not_exist")
    assert report.is_valid is False
    assert len(report.valid_pairs) == 0


# =========================================================================
# 2. CHECKPOINT LOADING & MODEL REGISTRY TESTS
# =========================================================================

def test_load_all_trained_checkpoints():
    """
    Verifies that all 3 trained checkpoints (SegFormer, U-Net, LinkNet)
    load their weights successfully without shape or key errors.
    """
    checkpoints = {
        "segformer": "ml/checkpoints/segformer_best.pt",
        "unet": "ml/checkpoints/unet_best.pt",
        "linknet": "ml/checkpoints/linknet_best.pt"
    }

    for name, ckpt_path in checkpoints.items():
        assert os.path.isfile(ckpt_path), f"Checkpoint missing: {ckpt_path}"
        model = create_model(name, device="cpu", checkpoint_path=ckpt_path, eval_mode=True)
        assert model is not None
        assert not model.training  # Must be in eval mode
        params = count_parameters(model)
        assert params > 1_000_000, f"Model {name} parameter count unexpectedly small: {params}"


def test_create_model_without_checkpoint():
    """
    Verifies model creation with random initial weights when checkpoint is omitted.
    """
    model = create_model("segformer", device="cpu", checkpoint_path=None, eval_mode=True)
    assert model is not None
    assert not model.training


def test_create_model_invalid_name():
    """
    Verifies ValueError is raised when requesting an unknown model name.
    """
    try:
        create_model("resnet_mask_rcnn")
        assert False, "Should have raised ValueError"
    except ValueError as ve:
        assert "Unknown model" in str(ve)


def test_model_size_mb_calculation():
    """
    Verifies calculation of model memory footprint in MB.
    """
    model = create_model("segformer", device="cpu")
    size_mb = get_model_size_mb(model)
    assert 10.0 <= size_mb <= 25.0


# =========================================================================
# 3. EXTENDED API TESTS
# =========================================================================

def test_samples_endpoint():
    """
    Verifies GET /api/samples returns all 3 pre-defined clinical cases.
    """
    response = client.get("/api/samples")
    assert response.status_code == 200
    samples = response.json()
    assert len(samples) == 3
    ids = [s["id"] for s in samples]
    assert "sessile" in ids
    assert "elevated" in ids
    assert "normal" in ids
    for s in samples:
        assert s["image_data_uri"].startswith("data:image/jpeg;base64,")
        assert s["mask_data_uri"].startswith("data:image/png;base64,")


def test_predict_invalid_model():
    """
    Verifies POST /api/predict returns 400 when an invalid model name is submitted.
    """
    import io
    img_arr = np.zeros((64, 64, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img_arr).save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    response = client.post(
        "/api/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"model_name": "non_existent_model"}
    )
    assert response.status_code == 400
    assert "Unknown model" in response.json()["detail"]


def test_predict_threshold_clamping():
    """
    Verifies POST /api/predict accepts extreme threshold inputs and clamps safely.
    """
    import io
    img_arr = np.zeros((64, 64, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img_arr).save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    response = client.post(
        "/api/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"model_name": "segformer", "threshold": "0.01"}
    )
    assert response.status_code == 200
    # Threshold clamped to 0.05
    assert response.json()["confidence_threshold"] == 0.05


def test_train_stop_when_idle():
    """
    Verifies POST /api/train/stop returns 400 when no training job is currently active.
    """
    status_resp = client.get("/api/train/status")
    if status_resp.json()["status"] != "running":
        response = client.post("/api/train/stop")
        assert response.status_code == 400
        assert "No training" in response.json()["detail"]
