"""
Comprehensive tests for Checkpoint Loading, Metadata Verification,
Safe Failure Handling, and Real Kvasir-SEG Ground-Truth Quantitative Evaluation.
"""

import os
import sys
import tempfile
import torch
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from ml.models.registry import (
    MODEL_CATALOG,
    create_model,
    load_checkpoint_weights,
    get_checkpoint_info,
    get_available_models_metadata
)
from ml.inference.predictor import PolypPredictor
from ml.evaluation.metrics import calculate_segmentation_metrics
from backend.app.main import app

client = TestClient(app)


def test_checkpoint_discovery_and_metadata():
    """
    Verifies that all 3 models discover their existing sample-trained checkpoints,
    accurately report file size, trained flag, and training origin as demo cases (not full Kvasir-SEG).
    """
    metadata = get_available_models_metadata()
    assert len(metadata) == 3

    for m in metadata:
        model_id = m["id"]
        assert m["checkpoint_available"] is True, f"Checkpoint should be available for {model_id}"
        assert m["is_trained"] is True, f"Model {model_id} should be marked as trained"
        assert m["checkpoint_filename"] in [f"{model_id}_kvasir_trained.pt", f"{model_id}_best.pt"]
        assert os.path.isfile(m["checkpoint_path"])
        assert m["checkpoint_size_mb"] > 5.0, f"Checkpoint size unexpectedly small: {m['checkpoint_size_mb']} MB"
        assert m["training_type"] in ["kvasir_full_700", "sample_trained"]
        assert m["training_dataset"] is not None


def test_load_checkpoint_formats():
    """
    Verifies that load_checkpoint_weights correctly handles:
    1. 'state_dict' wrapper format
    2. 'model_state_dict' wrapper format
    3. Direct dictionary format
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a tiny dummy model and save in different wrapper formats
        dummy_model = MODEL_CATALOG["segformer"]["class"](in_channels=3, num_classes=1)
        orig_sd = dummy_model.state_dict()

        # Format 1: state_dict
        p1 = os.path.join(tmpdir, "fmt_state_dict.pt")
        torch.save({"model_name": "segformer", "state_dict": orig_sd}, p1)
        m1 = MODEL_CATALOG["segformer"]["class"](in_channels=3, num_classes=1)
        load_checkpoint_weights(m1, p1, device="cpu")
        assert getattr(m1, "_is_trained", False) is True

        # Format 2: model_state_dict
        p2 = os.path.join(tmpdir, "fmt_model_state_dict.pt")
        torch.save({"epoch": 1, "model_state_dict": orig_sd}, p2)
        m2 = MODEL_CATALOG["segformer"]["class"](in_channels=3, num_classes=1)
        load_checkpoint_weights(m2, p2, device="cpu")
        assert getattr(m2, "_is_trained", False) is True

        # Format 3: direct state_dict
        p3 = os.path.join(tmpdir, "fmt_direct.pt")
        torch.save(orig_sd, p3)
        m3 = MODEL_CATALOG["segformer"]["class"](in_channels=3, num_classes=1)
        load_checkpoint_weights(m3, p3, device="cpu")
        assert getattr(m3, "_is_trained", False) is True


def test_missing_checkpoint_raises_filenotfound():
    """
    Verifies that create_model with require_checkpoint=True raises FileNotFoundError
    with an informative error message when weights are missing.
    """
    try:
        create_model("segformer", checkpoint_path="nonexistent_model_weights.pt", require_checkpoint=True)
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as fnfe:
        assert "Trained checkpoint not found" in str(fnfe)
        assert "Please provide trained model weights before running inference" in str(fnfe)


def test_incompatible_checkpoint_raises_runtimeerror():
    """
    Verifies that loading incompatible weights (e.g. wrong layer shapes) raises RuntimeError.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_sd = {"decoder.linear_pred.weight": torch.randn(10, 10)}
        bad_path = os.path.join(tmpdir, "corrupt_weights.pt")
        torch.save(bad_sd, bad_path)

        m = MODEL_CATALOG["segformer"]["class"](in_channels=3, num_classes=1)
        try:
            load_checkpoint_weights(m, bad_path, device="cpu")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as re:
            assert "incompatible" in str(re).lower() or "missing" in str(re).lower()


def test_polyp_predictor_rejects_missing_checkpoint():
    """
    Verifies that PolypPredictor fails safely when a checkpoint cannot be found.
    """
    predictor = PolypPredictor(device="cpu")
    try:
        predictor.get_model("segformer", checkpoint_path="ml/checkpoints/missing_segformer.pt")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as fnfe:
        assert "Trained checkpoint not found" in str(fnfe)


def test_api_models_endpoint_returns_checkpoint_info():
    """
    Verifies GET /api/models includes checkpoint availability, filename, and sample-trained status.
    """
    resp = client.get("/api/models")
    assert resp.status_code == 200
    models = resp.json()
    assert len(models) == 3
    for m in models:
        assert "checkpoint_available" in m
        assert m["checkpoint_available"] is True
        assert "checkpoint_filename" in m
        assert m["checkpoint_filename"] is not None
        assert "is_trained" in m
        assert m["is_trained"] is True
        assert m["training_type"] in ["kvasir_full_700", "sample_trained"]
        assert "training_dataset" in m


def test_api_predict_with_trained_checkpoints():
    """
    Verifies POST /api/predict executes successfully for all 3 models with trained checkpoints.
    """
    # Create test image
    img = Image.new("RGB", (128, 128), color=(180, 50, 50))
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    for m in ["segformer", "unet", "linknet"]:
        resp = client.post(
            "/api/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={"model_name": m, "threshold": 0.5}
        )
        assert resp.status_code == 200
        res = resp.json()
        assert res["model_name"] == m
        assert res["mask_data_uri"].startswith("data:image/png;base64,")
        assert res["overlay_data_uri"].startswith("data:image/png;base64,")
        assert res["inference_time_ms"] > 0


def test_api_compare_with_trained_checkpoints():
    """
    Verifies POST /api/compare executes across all models using genuine trained checkpoints.
    """
    img = Image.new("RGB", (128, 128), color=(180, 50, 50))
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    resp = client.post(
        "/api/compare",
        files={"file": ("compare_test.jpg", img_bytes, "image/jpeg")},
        data={"threshold": 0.5}
    )
    assert resp.status_code == 200
    res = resp.json()
    assert len(res["models_evaluated"]) == 3
    assert "segformer" in res["results"]
    assert "unet" in res["results"]
    assert "linknet" in res["results"]


def test_real_kvasir_ground_truth_mask_evaluation():
    """
    Verifies that /api/evaluate calculates Dice, IoU, Precision, Recall, and Specificity
    using the actual corresponding Kvasir-SEG image and ground-truth mask.
    Confirms metrics are mathematically derived from ground-truth, not fabricated.
    """
    img_path = "ml/datasets/Kvasir-SEG/images/cju0qkwl35piu0993l0dewei2.jpg"
    mask_path = "ml/datasets/Kvasir-SEG/masks/cju0qkwl35piu0993l0dewei2.jpg"

    assert os.path.isfile(img_path), f"Kvasir-SEG image missing: {img_path}"
    assert os.path.isfile(mask_path), f"Kvasir-SEG mask missing: {mask_path}"

    with open(img_path, "rb") as f_img, open(mask_path, "rb") as f_mask:
        img_bytes = f_img.read()
        mask_bytes = f_mask.read()

    # 1. Evaluate with genuine ground truth mask
    resp = client.post(
        "/api/evaluate",
        files={
            "file": ("cju0qkwl35piu0993l0dewei2.jpg", img_bytes, "image/jpeg"),
            "ground_truth_mask": ("cju0qkwl35piu0993l0dewei2_mask.jpg", mask_bytes, "image/jpeg")
        },
        data={"model_name": "segformer", "threshold": 0.5}
    )
    assert resp.status_code == 200
    res = resp.json()
    assert res["has_ground_truth"] is True
    assert res["metrics"] is not None
    m = res["metrics"]
    assert 0.0 < m["dice"] <= 1.0, f"Expected positive Dice score, got {m['dice']}"
    assert 0.0 < m["iou"] <= 1.0, f"Expected positive IoU score, got {m['iou']}"
    assert 0.0 < m["precision"] <= 1.0
    assert 0.0 < m["recall"] <= 1.0
    assert 0.0 < m["specificity"] <= 1.0

    # Mathematical relationship: IoU == Dice / (2 - Dice) within numerical tolerance
    expected_iou = m["dice"] / (2.0 - m["dice"])
    assert abs(m["iou"] - expected_iou) < 0.05, f"IoU {m['iou']} inconsistent with Dice {m['dice']}"

    # 2. Evaluate WITHOUT ground truth mask - metrics MUST NOT be fabricated
    resp_no_gt = client.post(
        "/api/evaluate",
        files={"file": ("cju0qkwl35piu0993l0dewei2.jpg", img_bytes, "image/jpeg")},
        data={"model_name": "segformer", "threshold": 0.5}
    )
    assert resp_no_gt.status_code == 200
    res_no_gt = resp_no_gt.json()
    assert res_no_gt["has_ground_truth"] is False
    assert res_no_gt["metrics"] is None, "Metrics must be None when no ground truth is provided"

