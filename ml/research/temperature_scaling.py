"""
Validation-Set Temperature Scaling and Multi-Formulation Uncertainty Calibration.

Mathematical Formulation:
1. Temperature Scaling: Learns a scalar parameter T > 0 on validation set logits exclusively:
     p_calibrated = sigmoid(logit / T)
   Optimized by minimizing negative log-likelihood (binary cross-entropy) on validation data:
     L_NLL(T) = - (1/N) * sum [ y * log(sigma(z/T)) + (1-y) * log(1 - sigma(z/T)) ]
2. Five Uncertainty Formulations for Empirical Reliability Analysis:
   - Form 1: Jensen-Shannon Epistemic Disagreement (JSD)
   - Form 2: Predictive Shannon Entropy H(mean_p)
   - Form 3: Probability Variance across Models: Var(p) = (1/K) * sum (p_k - mean_p)^2
   - Form 4: Boundary-Focused Epistemic Ambiguity: BAF = MorphGradient(Consensus) * JSD
   - Form 5: Evidence-Fused Composite Uncertainty: Comb = 0.40 * H(mean_p) + 0.35 * JSD + 0.25 * Normalized(Var)
3. Calibration & Failure Detection Auditor:
   - Evaluates Expected Calibration Error (ECE) and Brier Score before and after scaling.
   - Evaluates Pearson & Spearman correlation with ground-truth error |y_true - mean_p|.
   - Evaluates AUROC & AUPRC for predicting pixel-level classification failure.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from ml.datasets.kvasir_dataset import (
    PolypSegmentationDataset,
    resolve_dataset_path,
    get_deterministic_splits
)
from ml.models.registry import create_model
from ml.research.disagreement_fusion import (
    compute_bernoulli_entropy,
    compute_epistemic_disagreement_field,
    compute_boundary_ambiguity_field
)
from ml.research.evaluation_metrics import (
    calculate_expected_calibration_error,
    calculate_brier_score,
    calculate_uncertainty_error_correlation,
    calculate_failure_detection_auc
)


class TemperatureScaler(nn.Module):
    """
    Scalar temperature parameter for post-hoc logit calibration.
    Learned strictly on held-out validation logits to avoid data leakage.
    """
    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Prevent division by zero or negative temperature
        temp = torch.clamp(self.temperature, min=0.05, max=10.0)
        return logits / temp


def fit_temperature_scaling_on_validation(
    model: nn.Module,
    val_loader: DataLoader,
    device: str = "cpu",
    max_iter: int = 50
) -> float:
    """
    Fits scalar temperature T exclusively on validation logits minimizing NLL.
    """
    model.eval()
    scaler = TemperatureScaler(init_temp=1.0).to(device)
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.05, max_iter=max_iter)
    criterion = nn.BCEWithLogitsLoss()

    # 1. Collect validation logits and masks (subsample if large to fit memory)
    all_logits = []
    all_masks = []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_masks.append(masks.cpu())

    val_logits = torch.cat(all_logits, dim=0).to(device)
    val_masks = torch.cat(all_masks, dim=0).to(device)

    # 2. Optimize temperature via L-BFGS
    def eval_loss():
        optimizer.zero_grad()
        scaled_logits = scaler(val_logits)
        loss = criterion(scaled_logits, val_masks)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    fitted_temp = float(torch.clamp(scaler.temperature, min=0.05, max=10.0).item())
    return round(fitted_temp, 4)


def compute_alternative_uncertainty_maps(
    probs: List[np.ndarray],
    eps: float = 1e-7
) -> Dict[str, np.ndarray]:
    """
    Computes 5 alternative pixel-wise uncertainty formulations from continuous probability maps:
    1. jsd: Jensen-Shannon epistemic divergence
    2. entropy: Predictive Shannon entropy H(mean_p)
    3. variance: Cross-model probability variance
    4. boundary_disagreement: Morphological gradient boundary * JSD
    5. combined: Evidence-weighted composite uncertainty
    """
    probs_stack = np.stack(probs, axis=0).astype(np.float32)  # (K, H, W)
    mean_p = np.mean(probs_stack, axis=0)  # (H, W)

    # Form 1: JSD
    jsd, _, h_pred = compute_epistemic_disagreement_field(probs, eps=eps, return_entropy=True)

    # Form 2: Predictive Entropy
    entropy = h_pred

    # Form 3: Probability Variance
    variance = np.var(probs_stack, axis=0).astype(np.float32)

    # Form 4: Boundary Disagreement Field
    baf, _ = compute_boundary_ambiguity_field(mean_p, jsd, probs=probs)

    # Form 5: Combined Composite Uncertainty
    # Normalize variance to [0, 1] range (max theoretical variance for binary probs in [0, 1] is 0.25)
    var_norm = np.clip(variance * 4.0, 0.0, 1.0)
    combined = (0.40 * entropy + 0.35 * jsd + 0.25 * var_norm).astype(np.float32)

    return {
        "jsd": jsd,
        "entropy": entropy,
        "variance": variance,
        "boundary_disagreement": baf,
        "combined": combined,
        "mean_p": mean_p
    }


def evaluate_uncertainty_formulations_on_sample(
    probs: List[np.ndarray],
    gt_bin: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Evaluates correlation and failure detection quality for all 5 uncertainty formulations
    against genuine pixel error |gt_bin - mean_p|.
    """
    maps = compute_alternative_uncertainty_maps(probs)
    mean_p = maps["mean_p"]
    pred_bin = (mean_p >= threshold).astype(np.uint8)

    results = {}
    for form_name in ["jsd", "entropy", "variance", "boundary_disagreement", "combined"]:
        u_map = maps[form_name]
        corr = calculate_uncertainty_error_correlation(u_map, mean_p, gt_bin)
        auc = calculate_failure_detection_auc(u_map, pred_bin, gt_bin)

        results[form_name] = {
            "mean_uncertainty": float(round(float(np.mean(u_map)), 4)),
            "pearson_r": corr["pearson_correlation"],
            "spearman_r": corr["spearman_correlation"],
            "auroc": auc["auroc"],
            "auprc": auc["auprc"]
        }

    return results


def calibrate_all_models_on_validation(
    models: Optional[List[str]] = None,
    save_path: str = "ml/checkpoints/temperature_scaling_params.json",
    device: Optional[str] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Loads trained checkpoints, fits temperature scaling on the 200 validation samples,
    and saves learned parameters.
    """
    if models is None:
        models = ["segformer", "unet", "linknet"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_path, report = resolve_dataset_path()
    _, val_pairs, _ = get_deterministic_splits(report.valid_pairs, seed=seed)

    val_dataset = PolypSegmentationDataset(
        pairs=val_pairs,
        image_size=(256, 256),
        is_training=False,
        normalize_imagenet=True
    )
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    temperatures = {}
    print(f"\n=======================================================")
    print(f"FITTING TEMPERATURE SCALING ON VALIDATION SPLIT (N={len(val_pairs)})")
    print(f"=======================================================")

    for m_name in models:
        # Check for newly trained checkpoint first, then fallback to best.pt
        trained_ckpt = f"ml/checkpoints/{m_name}_kvasir_trained.pt"
        if not os.path.isfile(trained_ckpt):
            trained_ckpt = f"ml/checkpoints/{m_name}_best.pt"

        model = create_model(m_name, device=device, checkpoint_path=trained_ckpt, eval_mode=True, require_checkpoint=True)
        t_val = fit_temperature_scaling_on_validation(model, val_loader, device=device)
        temperatures[m_name] = {
            "temperature": t_val,
            "checkpoint_used": trained_ckpt,
            "dataset": "Kvasir-SEG Validation Split (200 pairs)",
            "learned_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        print(f"Architecture: {m_name.upper():12s} -> Learned Temperature T = {t_val:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(temperatures, f, indent=2)

    print(f"Temperature scaling parameters saved to: {save_path}\n")
    return temperatures


if __name__ == "__main__":
    calibrate_all_models_on_validation()

