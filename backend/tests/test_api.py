"""
Unit and integration tests for FastAPI Endpoints.
Verifies health, model metadata catalog, inference prediction, comparison, and validation rules.
"""

import io
import sys
import os
import pytest
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from backend.app.main import app

client = TestClient(app)


def create_test_image_bytes(width: int = 128, height: int = 128) -> bytes:
    img = Image.new("RGB", (width, height), color=(140, 40, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "registered_models" in data
    assert "segformer" in data["registered_models"]
    assert "device" in data
    assert data["dataset_present"] is True
    assert data["valid_image_mask_pairs"] == 1000


def test_models_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    models = response.json()
    assert len(models) == 3
    ids = [m["id"] for m in models]
    assert "segformer" in ids
    assert "unet" in ids
    assert "linknet" in ids


def test_experiments_endpoint():
    response = client.get("/api/experiments")
    assert response.status_code == 200
    records = response.json()
    assert len(records) >= 8
    # Check that both modern and historical models are present
    sources = set(r["source"] for r in records)
    assert any("Historical" in s for s in sources)
    assert any("Modern" in s for s in sources)


def test_dataset_status_endpoint():
    response = client.get("/api/dataset/status")
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is True
    assert data["dataset_available"] is True
    assert data["paired_count"] == 1000
    assert data["train_count"] == 700
    assert data["val_count"] == 200
    assert data["test_count"] == 100
    assert "Kvasir-SEG" in data["dataset_directory"]
    assert "instructions" in data


def test_predict_endpoint_valid():
    img_bytes = create_test_image_bytes()
    response = client.post(
        "/api/predict",
        files={"file": ("test_polyp.jpg", img_bytes, "image/jpeg")},
        data={"model_name": "segformer", "threshold": "0.5"}
    )
    assert response.status_code == 200
    res = response.json()
    assert res["model_name"] == "segformer"
    assert "mask_data_uri" in res
    assert res["mask_data_uri"].startswith("data:image/png;base64,")
    assert "overlay_data_uri" in res
    assert "inference_time_ms" in res
    assert res["inference_time_ms"] > 0
    assert "polyp_detected" in res
    assert "disclaimer" in res


def test_predict_endpoint_invalid_file_extension():
    fake_txt = b"this is not an image"
    response = client.post(
        "/api/predict",
        files={"file": ("notes.txt", fake_txt, "text/plain")},
        data={"model_name": "segformer"}
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_compare_endpoint():
    img_bytes = create_test_image_bytes()
    response = client.post(
        "/api/compare",
        files={"file": ("compare_frame.jpg", img_bytes, "image/jpeg")},
        data={"threshold": "0.5"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "models_evaluated" in data
    assert "results" in data
    for name in ["segformer", "unet", "linknet"]:
        assert name in data["results"]
        assert "overlay_data_uri" in data["results"][name]
