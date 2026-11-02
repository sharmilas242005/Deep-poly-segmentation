"""
Evaluation Metrics for Tri-Model Disagreement and Reliability Assessment.

Provides:
1. Standard Segmentation Metrics: Dice, IoU, Sensitivity, Specificity.
2. Calibration Metrics: Expected Calibration Error (ECE), Brier Score.
3. Uncertainty-Error Correlation (Pearson, Spearman).
4. Failure Detection Quality: AUROC and AUPRC for predicting pixel-level classification error.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def calculate_dice(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = 1e-7) -> float:
    """Computes Dice Similarity Coefficient between binary masks."""
    p = (pred_bin > 0).astype(np.float32)
    g = (gt_bin > 0).astype(np.float32)
    intersection = float(np.sum(p * g))
    cardinality = float(np.sum(p) + np.sum(g))
    if cardinality == 0:
        return 1.0
    return float((2.0 * intersection) / (cardinality + eps))


def calculate_iou(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = 1e-7) -> float:
    """Computes Jaccard Index (IoU) between binary masks."""
    p = (pred_bin > 0).astype(np.float32)
    g = (gt_bin > 0).astype(np.float32)
    intersection = float(np.sum(p * g))
    union = float(np.sum(p) + np.sum(g) - intersection)
    if union == 0:
        return 1.0
    return float(intersection / (union + eps))


def calculate_sensitivity(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = 1e-7) -> float:
    """Computes Sensitivity (Recall / True Positive Rate)."""
    p = (pred_bin > 0).astype(bool)
    g = (gt_bin > 0).astype(bool)
    tp = float(np.sum(p & g))
    fn = float(np.sum((~p) & g))
    if tp + fn == 0:
        return 1.0
    return float(tp / (tp + fn + eps))


def calculate_specificity(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = 1e-7) -> float:
    """Computes Specificity (True Negative Rate)."""
    p = (pred_bin > 0).astype(bool)
    g = (gt_bin > 0).astype(bool)
    tn = float(np.sum((~p) & (~g)))
    fp = float(np.sum(p & (~g)))
    if tn + fp == 0:
        return 1.0
    return float(tn / (tn + fp + eps))


def calculate_expected_calibration_error(
    probs: np.ndarray,
    gt_bin: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Computes Expected Calibration Error (ECE) with equal-width bins.
      ECE = sum_{m=1}^M ( |B_m| / N ) * | acc(B_m) - conf(B_m) |
    """
    p_flat = np.clip(probs.flatten().astype(np.float64), 0.0, 1.0)
    g_flat = (gt_bin.flatten() > 0).astype(np.float64)

    total_samples = len(p_flat)
    if total_samples == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (p_flat >= low) & (p_flat <= high)
        else:
            in_bin = (p_flat >= low) & (p_flat < high)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(g_flat[in_bin])
            bin_conf = np.mean(p_flat[in_bin])
            ece += (bin_count / total_samples) * np.abs(bin_acc - bin_conf)

    return float(round(ece, 4))


def calculate_brier_score(probs: np.ndarray, gt_bin: np.ndarray) -> float:
    """
    Computes mean squared error between continuous probabilities and ground truth:
      Brier = (1/N) * sum (p_i - y_i)^2
    """
    p_flat = np.clip(probs.flatten().astype(np.float64), 0.0, 1.0)
    g_flat = (gt_bin.flatten() > 0).astype(np.float64)
    if len(p_flat) == 0:
        return 0.0
    return float(round(np.mean((p_flat - g_flat) ** 2), 4))


def calculate_uncertainty_error_correlation(
    uncertainty_map: np.ndarray,
    probs: np.ndarray,
    gt_bin: np.ndarray
) -> Dict[str, float]:
    """
    Computes Pearson and Spearman correlation between prediction error |y_true - p|
    and uncertainty estimates (e.g., JSD or Predictive Entropy).
    """
    u_flat = uncertainty_map.flatten().astype(np.float64)
    p_flat = np.clip(probs.flatten().astype(np.float64), 0.0, 1.0)
    g_flat = (gt_bin.flatten() > 0).astype(np.float64)
    error_flat = np.abs(g_flat - p_flat)

    # Subsample if large to maintain speed
    if len(u_flat) > 20000:
        idx = np.random.RandomState(42).choice(len(u_flat), size=20000, replace=False)
        u_sub = u_flat[idx]
        e_sub = error_flat[idx]
    else:
        u_sub = u_flat
        e_sub = error_flat

    # Handle zero variance gracefully
    if np.std(u_sub) < 1e-9 or np.std(e_sub) < 1e-9:
        return {"pearson_correlation": 0.0, "spearman_correlation": 0.0}

    pearson_r, _ = stats.pearsonr(u_sub, e_sub)
    spearman_r, _ = stats.spearmanr(u_sub, e_sub)

    return {
        "pearson_correlation": float(round(pearson_r, 4)),
        "spearman_correlation": float(round(spearman_r, 4))
    }


def calculate_failure_detection_auc(
    uncertainty_map: np.ndarray,
    pred_bin: np.ndarray,
    gt_bin: np.ndarray
) -> Dict[str, float]:
    """
    Evaluates how effectively an uncertainty map flags binary pixel classification errors:
      y_error = 1 if pred_bin != gt_bin else 0
    Returns AUROC and AUPRC.
    """
    u_flat = uncertainty_map.flatten().astype(np.float64)
    p_bin = (pred_bin.flatten() > 0).astype(int)
    g_bin = (gt_bin.flatten() > 0).astype(int)
    y_error = (p_bin != g_bin).astype(int)

    n_errors = np.sum(y_error)
    total = len(y_error)

    # If completely accurate or completely inaccurate
    if n_errors == 0 or n_errors == total:
        return {"auroc": 1.0, "auprc": 1.0}

    # Subsample if extremely large
    if total > 50000:
        idx = np.random.RandomState(42).choice(total, size=50000, replace=False)
        u_eval = u_flat[idx]
        y_eval = y_error[idx]
    else:
        u_eval = u_flat
        y_eval = y_error

    if np.sum(y_eval) == 0 or np.sum(y_eval) == len(y_eval):
        return {"auroc": 1.0, "auprc": 1.0}

    try:
        auroc = float(roc_auc_score(y_eval, u_eval))
        prec, rec, _ = precision_recall_curve(y_eval, u_eval)
        auprc = float(auc(rec, prec))
    except Exception:
        auroc = 0.5
        auprc = float(n_errors / total)

    return {
        "auroc": float(round(auroc, 4)),
        "auprc": float(round(auprc, 4))
    }

