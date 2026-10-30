"""
Strict Mathematical Evaluation Metrics for Medical Image Segmentation.

Implements mathematically sound, unit-tested implementations of:
- Dice Similarity Coefficient (DSC) / F1 Score
- Intersection over Union (IoU) / Jaccard Index
- Precision / Positive Predictive Value (PPV)
- Recall / Sensitivity / True Positive Rate (TPR)
- Specificity / True Negative Rate (TNR)
- Pixel Accuracy
- Pixel Confusion Matrix Counts (TP, FP, FN, TN)

Mathematical Equivalences in Binary Segmentation:
1. Dice == Sorensen-Dice == F1 Score:
   DSC = (2 * TP) / (2 * TP + FP + FN)
   F1  = 2 * (Precision * Recall) / (Precision + Recall)
   DSC == F1 identically.
2. IoU == Jaccard Index:
   IoU = TP / (TP + FP + FN) == DSC / (2 - DSC)
"""

from typing import Dict, Union, Tuple, List
import numpy as np


def compute_confusion_matrix_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Tuple[int, int, int, int]:
    """
    Computes (TP, FP, FN, TN) counts at the pixel level.

    Args:
        y_true: Ground truth binary mask (H, W) or (B, H, W), float/int/bool
        y_pred: Predicted probability or binary mask (H, W) or (B, H, W)
        threshold: Decision threshold for turning probabilities into binary mask

    Returns:
        (tp, fp, fn, tn) as integer counts
    """
    # Ensure binary arrays
    true_bin = (np.asarray(y_true) >= 0.5).astype(bool)
    pred_bin = (np.asarray(y_pred) >= threshold).astype(bool)

    if true_bin.shape != pred_bin.shape:
        raise ValueError(f"Shape mismatch: y_true {true_bin.shape} != y_pred {pred_bin.shape}")

    tp = int(np.sum(true_bin & pred_bin))
    fp = int(np.sum(~true_bin & pred_bin))
    fn = int(np.sum(true_bin & ~pred_bin))
    tn = int(np.sum(~true_bin & ~pred_bin))

    return tp, fp, fn, tn


def calculate_segmentation_metrics(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    threshold: float = 0.5,
    eps: float = 1e-7
) -> Dict[str, float]:
    """
    Computes all standard medical image segmentation metrics for a single image or batch.

    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted mask (probabilities or binary)
        threshold: Classification threshold
        eps: Small epsilon to prevent division by zero while preserving numerical stability

    Returns:
        Dictionary of computed metrics with full precision.
    """
    tp, fp, fn, tn = compute_confusion_matrix_counts(y_true, y_pred, threshold)
    total_pixels = tp + fp + fn + tn

    # Edge case: Empty mask handling (e.g. healthy tissue with no polyp)
    is_true_empty = (tp + fn) == 0
    is_pred_empty = (tp + fp) == 0

    if is_true_empty and is_pred_empty:
        # Both ground truth and prediction contain no polyp pixels -> perfect true negative
        dice = 1.0
        iou = 1.0
        precision = 1.0
        recall = 1.0
        specificity = 1.0
        f1 = 1.0
    elif is_true_empty and not is_pred_empty:
        # Ground truth has no polyp, but model predicted false alarms
        dice = 0.0
        iou = 0.0
        precision = 0.0
        recall = 1.0  # Or 0.0 depending on clinical definition; here 0 positives to retrieve
        specificity = (tn) / (tn + fp + eps)
        f1 = 0.0
    elif not is_true_empty and is_pred_empty:
        # Ground truth has a polyp, but model missed it entirely
        dice = 0.0
        iou = 0.0
        precision = 0.0
        recall = 0.0
        specificity = 1.0
        f1 = 0.0
    else:
        dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        specificity = (tn + eps) / (tn + fp + eps)
        f1 = (2.0 * precision * recall + eps) / (precision + recall + eps)

    accuracy = (tp + tn) / max(total_pixels, 1)

    return {
        "dice": float(np.clip(dice, 0.0, 1.0)),
        "iou": float(np.clip(iou, 0.0, 1.0)),
        "precision": float(np.clip(precision, 0.0, 1.0)),
        "recall": float(np.clip(recall, 0.0, 1.0)),
        "sensitivity": float(np.clip(recall, 0.0, 1.0)),  # Sensitivity is identical to Recall
        "specificity": float(np.clip(specificity, 0.0, 1.0)),
        "f1_score": float(np.clip(f1, 0.0, 1.0)),
        "accuracy": float(np.clip(accuracy, 0.0, 1.0)),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "total_pixels": total_pixels,
        "polyp_area_fraction": float((tp + fn) / max(total_pixels, 1)),
        "predicted_area_fraction": float((tp + fp) / max(total_pixels, 1))
    }


def aggregate_dataset_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Computes mean and standard deviation across a list of per-image metric records.
    """
    if not metrics_list:
        return {}

    keys_to_average = [
        "dice", "iou", "precision", "recall", "sensitivity", "specificity", "f1_score", "accuracy"
    ]
    summary = {}
    for key in keys_to_average:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            summary[f"mean_{key}"] = float(np.mean(values))
            summary[f"std_{key}"] = float(np.std(values))
            summary[f"median_{key}"] = float(np.median(values))

    # Pixel-level global pooling (micro-average)
    total_tp = sum(m.get("true_positives", 0) for m in metrics_list)
    total_fp = sum(m.get("false_positives", 0) for m in metrics_list)
    total_fn = sum(m.get("false_negatives", 0) for m in metrics_list)
    total_tn = sum(m.get("true_negatives", 0) for m in metrics_list)

    eps = 1e-7
    summary["global_dice"] = float((2.0 * total_tp + eps) / (2.0 * total_tp + total_fp + total_fn + eps))
    summary["global_iou"] = float((total_tp + eps) / (total_tp + total_fp + total_fn + eps))
    summary["global_precision"] = float((total_tp + eps) / (total_tp + total_fp + eps))
    summary["global_recall"] = float((total_tp + eps) / (total_tp + total_fn + eps))
    summary["sample_count"] = len(metrics_list)

    return summary
