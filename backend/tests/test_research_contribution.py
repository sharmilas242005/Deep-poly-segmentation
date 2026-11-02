"""
Unit and Integration Tests for Research Contribution:
Multi-Architecture Epistemic Disagreement & Evidence-Calibrated Reliability Auditing (MED-ECRA).
"""

import os
import sys
import io
import json
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from backend.app.main import app
from ml.research.boundary_metrics import calculate_hd95, calculate_boundary_iou
from ml.research.stress_perturbations import (
    inject_specular_glare,
    inject_motion_blur,
    inject_fluid_turbidity
)
from ml.research.disagreement_fusion import (
    compute_bernoulli_entropy,
    compute_epistemic_disagreement_field,
    compute_boundary_ambiguity_field,
    fuse_predictions_evidence_calibrated,
    calculate_clinical_failure_risk_index
)

client = TestClient(app)


def test_boundary_metrics_computation():
    """Verifies that HD95 and Boundary IoU calculate correctly on controlled masks."""
    # 1. Identical squares: HD95 should be 0.0, BIoU should be 1.0
    m1 = np.zeros((100, 100), dtype=np.uint8)
    m1[30:70, 30:70] = 1
    m2 = m1.copy()

    assert calculate_hd95(m1, m2) == 0.0
    assert calculate_boundary_iou(m1, m2) == 1.0

    # 2. Shifted square: HD95 should reflect boundary offset
    m3 = np.zeros((100, 100), dtype=np.uint8)
    m3[35:75, 35:75] = 1

    hd95 = calculate_hd95(m1, m3)
    assert 4.0 <= hd95 <= 10.0
    biou = calculate_boundary_iou(m1, m3)
    assert 0.0 < biou < 1.0

    # 3. Empty mask handling
    empty = np.zeros((100, 100), dtype=np.uint8)
    assert calculate_hd95(empty, empty) == 0.0
    assert calculate_boundary_iou(empty, empty) == 1.0


def test_stress_perturbations_integrity():
    """Verifies that controlled endoscopic stress perturbations maintain shape and valid uint8 bounds."""
    test_img = np.full((128, 128, 3), 120, dtype=np.uint8)

    # 1. Specular glare
    glare = inject_specular_glare(test_img, num_spots=3, max_radius=12)
    assert glare.shape == (128, 128, 3)
    assert glare.dtype == np.uint8
    assert np.max(glare) > 120  # Specular highlights increase luminance

    # 2. Motion blur
    blur = inject_motion_blur(test_img, kernel_size=11, angle_degrees=45.0)
    assert blur.shape == (128, 128, 3)
    assert blur.dtype == np.uint8

    # 3. Fluid turbidity
    turbid = inject_fluid_turbidity(test_img)
    assert turbid.shape == (128, 128, 3)
    assert turbid.dtype == np.uint8


def test_epistemic_disagreement_and_fusion():
    """Verifies mathematical properties of JSD entropy and Evidence-Calibrated fusion."""
    # When all 3 models completely agree (p = 0.9 everywhere)
    p_agree = [np.full((32, 32), 0.9, dtype=np.float32) for _ in range(3)]
    jsd_agree, mean_agree = compute_epistemic_disagreement_field(p_agree)
    assert np.allclose(jsd_agree, 0.0, atol=1e-5), "JSD must be 0 when models agree"

    # When models completely disagree (one says 0.95, two say 0.05)
    p_disagree = [
        np.full((32, 32), 0.95, dtype=np.float32),
        np.full((32, 32), 0.05, dtype=np.float32),
        np.full((32, 32), 0.05, dtype=np.float32)
    ]
    jsd_disagree, _ = compute_epistemic_disagreement_field(p_disagree)
    assert np.mean(jsd_disagree) > 0.15, "JSD must be positive under architectural discordance"

    # Fusion
    fused = fuse_predictions_evidence_calibrated(p_disagree)
    assert fused.shape == (32, 32)
    assert 0.0 <= np.min(fused) <= np.max(fused) <= 1.0

    # CFRI calculation
    baf, bdr = compute_boundary_ambiguity_field(mean_agree, jsd_agree)
    cfri, level = calculate_clinical_failure_risk_index(jsd_disagree, bdr, p_disagree)
    assert 0.0 <= cfri <= 100.0
    assert isinstance(level, str)


def test_api_research_analyze_endpoint():
    """Verifies POST /api/research/analyze executes and returns all diagnostic fields."""
    img = Image.new("RGB", (128, 128), color=(140, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    resp = client.post(
        "/api/research/analyze",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"threshold": 0.50}
    )
    assert resp.status_code == 200
    res = resp.json()

    assert "fused_mask_data_uri" in res
    assert res["fused_mask_data_uri"].startswith("data:image/png;base64,")
    assert "disagreement_heatmap_data_uri" in res
    assert "boundary_ambiguity_data_uri" in res
    assert "clinical_failure_risk_index" in res
    assert 0.0 <= res["clinical_failure_risk_index"] <= 100.0
    assert "risk_level" in res
    assert res["models_fused"] == ["segformer", "unet", "linknet"]
    assert res["inference_time_ms"] > 0


def test_api_research_analyze_with_ground_truth():
    """Verifies POST /api/research/analyze calculates HD95 and boundary metrics with real Kvasir-SEG mask."""
    img_path = "ml/datasets/Kvasir-SEG/images/cju0qkwl35piu0993l0dewei2.jpg"
    mask_path = "ml/datasets/Kvasir-SEG/masks/cju0qkwl35piu0993l0dewei2.jpg"

    assert os.path.isfile(img_path)
    assert os.path.isfile(mask_path)

    with open(img_path, "rb") as fi, open(mask_path, "rb") as fm:
        resp = client.post(
            "/api/research/analyze",
            files={
                "file": ("cju0qkwl35piu0993l0dewei2.jpg", fi.read(), "image/jpeg"),
                "ground_truth_mask": ("cju0qkwl35piu0993l0dewei2.jpg", fm.read(), "image/jpeg")
            },
            data={"threshold": 0.50}
        )
    assert resp.status_code == 200
    res = resp.json()
    assert res["has_ground_truth"] is True
    assert res["fused_metrics"] is not None
    assert "hd95" in res["fused_metrics"]
    assert "biou" in res["fused_metrics"]
    assert "dice" in res["fused_metrics"]
    assert "model_metrics" in res
    assert "segformer" in res["model_metrics"]


def test_api_research_benchmarks_endpoint():
    """Verifies GET /api/research/benchmarks returns empirical test split results."""
    resp = client.get("/api/research/benchmarks")
    assert resp.status_code == 200
    data = resp.json()
    assert "benchmark_dataset" in data
    assert "clean_benchmarks" in data
    assert "stress_glare_benchmarks" in data
    assert "stress_blur_benchmarks" in data
    assert "scale_stratified_benchmarks" in data

