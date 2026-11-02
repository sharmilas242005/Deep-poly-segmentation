"""
Comprehensive Scientific and Technical Test Suite for
Tri-Model Disagreement-Guided Polyp Segmentation Reliability Assessment.
"""

import os
import io
import pytest
import numpy as np
import torch
from PIL import Image
from fastapi.testclient import TestClient

from backend.app.main import app
from ml.models.registry import create_model, get_checkpoint_info, MODEL_CATALOG
from ml.inference.predictor import PolypPredictor
from ml.research.disagreement_fusion import (
    compute_bernoulli_entropy,
    compute_epistemic_disagreement_field,
    compute_boundary_ambiguity_field,
    fuse_predictions_evidence_calibrated,
    calculate_model_disagreement_indicator,
    calculate_clinical_failure_risk_index,
    run_tri_model_disagreement_analysis
)
from ml.research.evaluation_metrics import (
    calculate_dice,
    calculate_iou,
    calculate_sensitivity,
    calculate_specificity,
    calculate_expected_calibration_error,
    calculate_brier_score,
    calculate_uncertainty_error_correlation,
    calculate_failure_detection_auc
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    # Create valid synthetic RGB endoscopy frame
    arr = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_mask_bytes():
    # Create synthetic binary lesion mask
    arr = np.zeros((256, 256), dtype=np.uint8)
    arr[64:192, 64:192] = 255  # Square lesion
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Model Verification and Checkpoint Loading
# ---------------------------------------------------------------------------

def test_all_three_models_loaded_with_trained_checkpoints():
    """Confirms U-Net, LinkNet, SegFormer-B0 load genuine checkpoints in eval mode."""
    for model_name in ["segformer", "unet", "linknet"]:
        info = get_checkpoint_info(model_name)
        assert info["checkpoint_available"] is True, f"Checkpoint missing for {model_name}"
        assert info["is_trained"] is True, f"Model {model_name} must be flagged as trained"
        assert info["training_type"] in ["kvasir_full_700", "sample_trained"]

        model = create_model(model_name, eval_mode=True, require_checkpoint=True)
        assert model.training is False, f"Model {model_name} must be in eval mode"
        assert getattr(model, "_is_trained", False) is True

        # Forward pass on normalized tensor
        dummy = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            logits = model(dummy)
            probs = torch.sigmoid(logits)

        assert logits.shape == (1, 1, 256, 256)
        assert not torch.isnan(probs).any(), f"NaN detected in {model_name} output"
        assert not torch.isinf(probs).any(), f"Inf detected in {model_name} output"
        assert probs.min().item() >= 0.0 and probs.max().item() <= 1.0


def test_polyp_predictor_continuous_probability_output(sample_image_bytes):
    """Verifies PolypPredictor returns continuous probability arrays [0, 1] without thresholding."""
    predictor = PolypPredictor(device="cpu")
    res, prob_map = predictor.predict(
        image_input=sample_image_bytes,
        model_name="segformer",
        threshold=0.50,
        return_raw_prob_map=True
    )

    assert isinstance(prob_map, np.ndarray)
    assert prob_map.dtype == np.float32
    assert prob_map.min() >= 0.0 and prob_map.max() <= 1.0
    # Must contain continuous gradations, not just binary 0 and 1
    unique_vals = len(np.unique(prob_map))
    assert unique_vals > 10, f"Expected continuous distribution, got only {unique_vals} unique values"
    assert "probability_map_data_uri" in res
    assert res["probability_map_data_uri"].startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# 2. Mathematical Stability of Entropy & Epistemic Disagreement (JSD)
# ---------------------------------------------------------------------------

def test_bernoulli_entropy_numerical_stability():
    """Tests binary entropy near extremes 0.0 and 1.0 and at peak 0.5."""
    p_extremes = np.array([0.0, 1e-12, 0.5, 1.0 - 1e-12, 1.0], dtype=np.float32)
    h = compute_bernoulli_entropy(p_extremes)

    assert not np.isnan(h).any()
    assert not np.isinf(h).any()
    # At p=0 and p=1, entropy approaches 0
    assert h[0] < 1e-4
    assert h[4] < 1e-4
    # At p=0.5, binary Shannon entropy with log2 is exactly 1.0
    assert np.isclose(h[2], 1.0, atol=1e-3)


def test_epistemic_disagreement_jsd_properties():
    """Tests Jensen-Shannon Divergence calculation across probability maps."""
    H, W = 64, 64
    # Case A: Identical models -> JSD must be exactly 0
    p_identical = [np.full((H, W), 0.75, dtype=np.float32) for _ in range(3)]
    jsd_ident, mean_p, h_pred = compute_epistemic_disagreement_field(p_identical, return_entropy=True)
    assert np.allclose(jsd_ident, 0.0, atol=1e-5)
    assert np.allclose(mean_p, 0.75, atol=1e-5)

    # Case B: Maximum discordance (two models predict 1.0, one predicts 0.0)
    p_conflict = [
        np.full((H, W), 1.0, dtype=np.float32),
        np.full((H, W), 1.0, dtype=np.float32),
        np.full((H, W), 0.0, dtype=np.float32)
    ]
    jsd_conf, mean_conf, h_pred_conf = compute_epistemic_disagreement_field(p_conflict, return_entropy=True)
    assert np.all(jsd_conf > 0.50), "High cross-model discordance must yield high JSD"
    assert np.all(jsd_conf <= 1.0)
    assert np.all(h_pred_conf >= jsd_conf), "Predictive entropy must be an upper bound on JSD"


# ---------------------------------------------------------------------------
# 3. Boundary Ambiguity and Empty Consensus Handling
# ---------------------------------------------------------------------------

def test_boundary_ambiguity_calculation_and_empty_consensus():
    """Verifies perimeter isolation and prevents false zero-uncertainty reporting on empty consensus."""
    H, W = 100, 100
    mean_p = np.zeros((H, W), dtype=np.float32)
    mean_p[30:70, 30:70] = 0.85
    jsd = np.full((H, W), 0.40, dtype=np.float32)

    baf, bdr = compute_boundary_ambiguity_field(mean_p, jsd, dilation_radius=2)
    assert baf.shape == (H, W)
    assert bdr > 0.0, "Lesion with clear perimeter must have positive BDR"

    # Empty consensus test with conflicting models:
    # SegFormer predicts 0.45, UNet predicts 0.10, LinkNet predicts 0.10 -> mean_p is ~0.216 (< 0.50 threshold)
    p1 = np.zeros((H, W), dtype=np.float32)
    p1[30:70, 30:70] = 0.45
    p2 = np.zeros((H, W), dtype=np.float32)
    p3 = np.zeros((H, W), dtype=np.float32)
    probs_empty = [p1, p2, p3]
    jsd_empty, mean_empty, _ = compute_epistemic_disagreement_field(probs_empty, return_entropy=True)

    baf_empty, bdr_empty = compute_boundary_ambiguity_field(mean_empty, jsd_empty, probs=probs_empty)
    # Must NOT falsely report zero boundary disagreement when one model detects a candidate lesion
    assert bdr_empty > 0.0, "Conflicting models without hard consensus must not report zero disagreement"


# ---------------------------------------------------------------------------
# 4. Consensus Fusion (ECCF) and Disagreement Indicator
# ---------------------------------------------------------------------------

def test_eccf_evidence_weighted_fusion():
    """Verifies heuristic evidence-weighted fusion penalizes distance to consensus."""
    H, W = 50, 50
    # Two models agree on 0.9, one outlier predicts 0.1
    p1 = np.full((H, W), 0.9, dtype=np.float32)
    p2 = np.full((H, W), 0.9, dtype=np.float32)
    p3 = np.full((H, W), 0.1, dtype=np.float32)

    fused = fuse_predictions_evidence_calibrated([p1, p2, p3], sigma_w=0.25)
    # Fused result should be pulled closer to 0.9 than a naive arithmetic mean of 0.633
    arithmetic_mean = (0.9 + 0.9 + 0.1) / 3.0
    assert np.mean(fused) > arithmetic_mean, "ECCF should downweight the distant outlier"
    assert fused.min() >= 0.0 and fused.max() <= 1.0

    # Test invalid sigma_w
    with pytest.raises(ValueError):
        fuse_predictions_evidence_calibrated([p1, p2, p3], sigma_w=0.0)


def test_model_disagreement_indicator_non_clinical():
    """Verifies non-clinical disagreement score and heuristic levels."""
    H, W = 50, 50
    jsd_low = np.full((H, W), 0.02, dtype=np.float32)
    p_concordant = [np.full((H, W), 0.8, dtype=np.float32) for _ in range(3)]
    score_low, level_low = calculate_model_disagreement_indicator(jsd_low, 0.02, p_concordant)
    assert score_low < 20.0
    assert "High Consensus" in level_low

    # Backward compatibility alias test
    alias_score, alias_level = calculate_clinical_failure_risk_index(jsd_low, 0.02, p_concordant)
    assert alias_score == score_low
    assert alias_level == level_low


# ---------------------------------------------------------------------------
# 5. Scientific Evaluation Metrics
# ---------------------------------------------------------------------------

def test_research_evaluation_metrics():
    """Verifies Dice, IoU, Sensitivity, Specificity, ECE, Brier, and AUROC."""
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[20:80, 20:80] = 1

    # Exact match
    pred_exact = gt.copy()
    assert np.isclose(calculate_dice(pred_exact, gt), 1.0)
    assert np.isclose(calculate_iou(pred_exact, gt), 1.0)
    assert np.isclose(calculate_sensitivity(pred_exact, gt), 1.0)
    assert np.isclose(calculate_specificity(pred_exact, gt), 1.0)

    # Completely disjoint
    pred_disjoint = np.zeros_like(gt)
    pred_disjoint[0:15, 0:15] = 1
    assert np.isclose(calculate_dice(pred_disjoint, gt), 0.0)
    assert np.isclose(calculate_iou(pred_disjoint, gt), 0.0)

    # Calibration metrics
    probs_perfect = gt.astype(np.float32)
    assert np.isclose(calculate_brier_score(probs_perfect, gt), 0.0)
    assert np.isclose(calculate_expected_calibration_error(probs_perfect, gt), 0.0)

    # Failure detection AUROC
    uncertainty = np.random.rand(100, 100).astype(np.float32)
    auc_res = calculate_failure_detection_auc(uncertainty, pred_exact, gt)
    assert "auroc" in auc_res and "auprc" in auc_res

    # Uncertainty error correlation
    corr_res = calculate_uncertainty_error_correlation(uncertainty, probs_perfect, gt)
    assert "pearson_correlation" in corr_res
    assert "spearman_correlation" in corr_res


# ---------------------------------------------------------------------------
# 6. Tri-Model Disagreement API Endpoint
# ---------------------------------------------------------------------------

def test_disagreement_analysis_api_endpoint(client, sample_image_bytes):
    """Tests POST /api/segmentation/disagreement-analysis with real models."""
    response = client.post(
        "/api/segmentation/disagreement-analysis",
        data={"threshold": 0.50, "sigma_w": 0.25},
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    )

    assert response.status_code == 200
    data = response.json()

    assert data["models_evaluated"] == ["segformer", "unet", "linknet"]
    assert "segformer" in data["model_predictions"]
    assert "unet" in data["model_predictions"]
    assert "linknet" in data["model_predictions"]

    # Verify each model returned genuine prediction data
    for m in ["segformer", "unet", "linknet"]:
        m_data = data["model_predictions"][m]
        assert m_data["inference_time_ms"] > 0
        assert m_data["mask_data_uri"].startswith("data:image/png;base64,")
        assert m_data["probability_map_data_uri"].startswith("data:image/png;base64,")

    # Verify tri-model fusion and uncertainty maps
    assert data["mean_probability_map_data_uri"].startswith("data:image/png;base64,")
    assert data["consensus_mask_data_uri"].startswith("data:image/png;base64,")
    assert data["consensus_overlay_data_uri"].startswith("data:image/png;base64,")
    assert data["jsd_heatmap_data_uri"].startswith("data:image/png;base64,")
    assert data["predictive_entropy_heatmap_data_uri"].startswith("data:image/png;base64,")
    assert data["boundary_disagreement_heatmap_data_uri"].startswith("data:image/png;base64,")

    # Verify metrics and latencies
    assert 0.0 <= data["model_disagreement_indicator"] <= 100.0
    assert isinstance(data["reliability_level"], str)
    assert data["mean_jsd"] >= 0.0
    assert data["total_inference_time_ms"] > 0
    assert data["fusion_time_ms"] is not None


def test_disagreement_analysis_api_with_ground_truth(client, sample_image_bytes, sample_mask_bytes):
    """Tests POST /api/segmentation/disagreement-analysis with ground truth evaluation."""
    response = client.post(
        "/api/segmentation/disagreement-analysis",
        data={"threshold": 0.50, "sigma_w": 0.25},
        files={
            "file": ("test.jpg", sample_image_bytes, "image/jpeg"),
            "ground_truth_mask": ("mask.png", sample_mask_bytes, "image/png")
        }
    )

    assert response.status_code == 200
    data = response.json()
    eval_data = data.get("evaluation_metrics")
    assert eval_data is not None
    assert eval_data["status"] == "completed"

    # Verify all 6 comparison configurations are evaluated
    expected_configs = ["segformer", "unet", "linknet", "majority_voting", "mean_probability_fusion", "eccf_fusion"]
    for cfg in expected_configs:
        assert cfg in eval_data["method_comparison"]
        m = eval_data["method_comparison"][cfg]
        assert "dice" in m and "iou" in m and "sensitivity" in m and "specificity" in m
        assert "brier_score" in m and "expected_calibration_error" in m


def test_disagreement_analysis_invalid_inputs(client, sample_image_bytes):
    """Tests error handling for invalid files and parameters."""
    # Unsupported extension
    resp_ext = client.post(
        "/api/segmentation/disagreement-analysis",
        files={"file": ("test.pdf", b"%PDF-1.4...", "application/pdf")}
    )
    assert resp_ext.status_code == 400

    # Invalid sigma_w <= 0
    resp_sigma = client.post(
        "/api/segmentation/disagreement-analysis",
        data={"sigma_w": -0.5},
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    )
    assert resp_sigma.status_code == 400
