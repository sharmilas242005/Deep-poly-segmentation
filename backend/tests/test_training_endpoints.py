import os
import sys
import time
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from backend.app.main import app
from backend.app.services.training_manager import training_manager

client = TestClient(app)


def test_train_status_initial():
    response = client.get("/api/train/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["idle", "completed", "stopped", "running"]


def test_train_start_and_status_tracking():
    # Start training for 2 epochs on sample_images
    response = client.post(
        "/api/train/start",
        json={
            "model_name": "segformer",
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 0.001,
            "dataset_source": "sample_images",
            "early_stopping_patience": 3
        }
    )
    assert response.status_code == 200, f"Start failed: {response.text}"
    start_data = response.json()
    assert start_data["success"] is True

    # Poll status until completed or timed out (max 30 seconds)
    completed = False
    for _ in range(60):
        time.sleep(0.5)
        status_resp = client.get("/api/train/status")
        assert status_resp.status_code == 200
        st = status_resp.json()
        if st["status"] == "completed":
            completed = True
            assert st["current_epoch"] == 2
            assert len(st["history"]) == 2
            assert st["train_loss"] is not None
            assert st["val_dice"] is not None
            assert st["learning_rate"] is not None
            assert st["best_dice"] is not None
            break
        elif st["status"] == "failed":
            pytest.fail(f"Training worker failed: {st.get('error_message')}")

    assert completed, "Training did not complete within timeout."


def test_evaluate_endpoint_with_ground_truth():
    img_path = "data/sample_images/sample_polyp_sessile.jpg"
    mask_path = "data/sample_images/sample_polyp_sessile_mask.png"

    with open(img_path, "rb") as f_img, open(mask_path, "rb") as f_mask:
        response = client.post(
            "/api/evaluate",
            files={
                "file": ("sessile.jpg", f_img.read(), "image/jpeg"),
                "ground_truth_mask": ("mask.png", f_mask.read(), "image/png")
            },
            data={"model_name": "segformer", "threshold": 0.5}
        )

    assert response.status_code == 200, f"Evaluate failed: {response.text}"
    data = response.json()
    assert data["has_ground_truth"] is True
    assert data["metrics"] is not None
    assert "dice" in data["metrics"]
    assert "iou" in data["metrics"]
    assert "precision" in data["metrics"]
    assert "recall" in data["metrics"]
    assert "specificity" in data["metrics"]
    assert data["ground_truth_mask_data_uri"] is not None
    assert 0.0 <= data["metrics"]["dice"] <= 1.0
    assert data["metrics"]["dice"] > 0.5
    assert 0.0 <= data["metrics"]["iou"] <= 1.0
    assert 0.0 <= data["metrics"]["specificity"] <= 1.0


def test_evaluate_endpoint_without_ground_truth():
    img_path = "data/sample_images/sample_polyp_sessile.jpg"

    with open(img_path, "rb") as f_img:
        response = client.post(
            "/api/evaluate",
            files={
                "file": ("sessile.jpg", f_img.read(), "image/jpeg")
            },
            data={"model_name": "segformer", "threshold": 0.5}
        )

    assert response.status_code == 200, f"Evaluate failed: {response.text}"
    data = response.json()
    assert data["has_ground_truth"] is False
    assert data["metrics"] is None
    assert data["ground_truth_mask_data_uri"] is None
