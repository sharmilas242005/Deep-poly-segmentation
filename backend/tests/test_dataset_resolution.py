"""
Tests for Dataset Path Resolution Priority and Kvasir-SEG Dataset Status.
Verifies:
1. Priority 1: DATASET_ROOT_DIR environment variable override when valid.
2. Priority 2: ml/datasets/Kvasir-SEG default discovery.
3. Fallback priority: data/sample_images.
4. Total image-mask pairs and deterministic split reproducibility.
5. Consistency between /api/dataset/status and training manager dataset resolution.
"""

import os
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from ml.datasets.kvasir_dataset import (
    resolve_dataset_path,
    validate_kvasir_dataset,
    get_deterministic_splits,
    PolypSegmentationDataset
)

client = TestClient(app)


def test_default_resolution_finds_kvasir_seg():
    # When DATASET_ROOT_DIR is not set, it must resolve ml/datasets/Kvasir-SEG
    orig_env = os.environ.pop("DATASET_ROOT_DIR", None)
    try:
        resolved_path, report = resolve_dataset_path()
        assert report.is_valid is True
        assert len(report.valid_pairs) == 1000
        assert "Kvasir-SEG" in resolved_path
    finally:
        if orig_env is not None:
            os.environ["DATASET_ROOT_DIR"] = orig_env


def test_env_var_priority_override(monkeypatch):
    # When DATASET_ROOT_DIR is set to a valid directory, it takes priority
    monkeypatch.setenv("DATASET_ROOT_DIR", "data/sample_images")
    resolved_path, report = resolve_dataset_path()
    assert report.is_valid is True
    assert len(report.valid_pairs) == 3
    assert "sample_images" in resolved_path


def test_invalid_env_var_falls_back_to_kvasir_seg(monkeypatch):
    # When DATASET_ROOT_DIR points to an invalid/nonexistent path, it falls back to ml/datasets/Kvasir-SEG
    monkeypatch.setenv("DATASET_ROOT_DIR", "non_existent_folder_xyz")
    resolved_path, report = resolve_dataset_path()
    assert report.is_valid is True
    assert len(report.valid_pairs) == 1000
    assert "Kvasir-SEG" in resolved_path


def test_dataset_splits_count():
    resolved_path, report = resolve_dataset_path()
    train_pairs, val_pairs, test_pairs = get_deterministic_splits(report.valid_pairs, seed=42)
    assert len(train_pairs) == 700
    assert len(val_pairs) == 200
    assert len(test_pairs) == 100
    assert len(train_pairs) + len(val_pairs) + len(test_pairs) == 1000


def test_training_manager_path_consistency():
    # Training manager when dataset_source is 'auto' must resolve the exact same dataset path as status endpoint
    status_response = client.get("/api/dataset/status")
    assert status_response.status_code == 200
    status_data = status_response.json()

    resolved_path, _ = resolve_dataset_path()
    assert os.path.normpath(status_data["dataset_directory"]) == os.path.normpath(os.path.abspath(resolved_path))
    assert status_data["paired_count"] == 1000
    assert status_data["train_count"] == 700
    assert status_data["val_count"] == 200
    assert status_data["test_count"] == 100


def test_dataset_loader_batch_generation():
    resolved_path, report = resolve_dataset_path()
    train_pairs, val_pairs, _ = get_deterministic_splits(report.valid_pairs, seed=42)
    ds = PolypSegmentationDataset(train_pairs[:2], image_size=(256, 256), is_training=False)
    sample = ds[0]
    assert sample["image"].shape == (3, 256, 256)
    assert sample["mask"].shape == (1, 256, 256)
    unique_vals = set(sample["mask"].unique().tolist())
    assert unique_vals.issubset({0.0, 1.0})

