"""
Unit tests for strict mathematical segmentation evaluation metrics.
Verifies Dice, IoU, Precision, Recall, Specificity, F1, and empty mask edge cases.
"""

import numpy as np
import pytest
import sys
import os

# Add root directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from ml.evaluation.metrics import (
    calculate_segmentation_metrics,
    compute_confusion_matrix_counts,
    aggregate_dataset_metrics
)


def test_perfect_overlap():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 20:60] = 1

    metrics = calculate_segmentation_metrics(mask, mask)
    assert metrics["dice"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["iou"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["precision"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["recall"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["sensitivity"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["specificity"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["f1_score"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["accuracy"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0


def test_zero_overlap():
    true_mask = np.zeros((50, 50), dtype=np.uint8)
    true_mask[0:20, 0:20] = 1

    pred_mask = np.zeros((50, 50), dtype=np.uint8)
    pred_mask[30:50, 30:50] = 1

    metrics = calculate_segmentation_metrics(true_mask, pred_mask)
    assert metrics["dice"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["iou"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["precision"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["recall"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["true_positives"] == 0
    assert metrics["false_positives"] == 400
    assert metrics["false_negatives"] == 400


def test_known_hand_calculated_overlap():
    # 10x10 grid = 100 pixels
    # True positives = 20
    # False positives = 10
    # False negatives = 20
    # True negatives = 50
    true_mask = np.zeros(100, dtype=np.uint8)
    pred_mask = np.zeros(100, dtype=np.uint8)

    true_mask[0:40] = 1  # 40 positive ground truth
    pred_mask[0:20] = 1  # 20 overlap (TP)
    pred_mask[40:50] = 1 # 10 false positive (FP)
    # Remaining 20 in true_mask[20:40] are false negatives (FN)
    # Remaining 50 in true_mask[50:100] are true negatives (TN)

    tp, fp, fn, tn = compute_confusion_matrix_counts(true_mask, pred_mask)
    assert tp == 20
    assert fp == 10
    assert fn == 20
    assert tn == 50

    metrics = calculate_segmentation_metrics(true_mask, pred_mask)
    expected_dice = 40.0 / 70.0  # 4/7 ~ 0.571428
    expected_iou = 20.0 / 50.0   # 0.40
    expected_precision = 20.0 / 30.0  # 2/3 ~ 0.666667
    expected_recall = 20.0 / 40.0     # 0.50
    expected_specificity = 50.0 / 60.0 # 5/6 ~ 0.833333

    assert metrics["dice"] == pytest.approx(expected_dice, abs=1e-4)
    assert metrics["iou"] == pytest.approx(expected_iou, abs=1e-4)
    assert metrics["precision"] == pytest.approx(expected_precision, abs=1e-4)
    assert metrics["recall"] == pytest.approx(expected_recall, abs=1e-4)
    assert metrics["specificity"] == pytest.approx(expected_specificity, abs=1e-4)
    assert metrics["f1_score"] == pytest.approx(expected_dice, abs=1e-4)


def test_dice_and_iou_mathematical_identity():
    # In binary segmentation: IoU = Dice / (2 - Dice)
    for dice_val in [0.2, 0.5, 0.75, 0.88, 0.95]:
        expected_iou = dice_val / (2.0 - dice_val)
        # Construct synthetic masks matching this ratio
        total_p = 1000
        # Dice = 2TP / (2TP + FP + FN) => 2TP / Union_sum
        # If FP=FN=k, 2TP / (2TP + 2k) = TP / (TP + k) = dice_val
        # k = TP*(1 - dice_val)/dice_val
        tp = 200
        k = int(round(tp * (1.0 - dice_val) / dice_val))
        true_m = np.zeros(tp + 2 * k + 100, dtype=np.uint8)
        pred_m = np.zeros(tp + 2 * k + 100, dtype=np.uint8)
        true_m[:tp + k] = 1
        pred_m[:tp] = 1
        pred_m[tp + k: tp + 2 * k] = 1

        m = calculate_segmentation_metrics(true_m, pred_m)
        computed_iou = m["iou"]
        relation_iou = m["dice"] / (2.0 - m["dice"])
        assert computed_iou == pytest.approx(relation_iou, abs=1e-3)


def test_empty_mask_handling():
    # Case 1: Healthy image with no polyp, model correctly predicts empty
    empty_true = np.zeros((64, 64), dtype=np.uint8)
    empty_pred = np.zeros((64, 64), dtype=np.uint8)
    m = calculate_segmentation_metrics(empty_true, empty_pred)
    assert m["dice"] == 1.0
    assert m["iou"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["specificity"] == 1.0

    # Case 2: Healthy image, but model has false positive
    fp_pred = np.zeros((64, 64), dtype=np.uint8)
    fp_pred[10:20, 10:20] = 1
    m2 = calculate_segmentation_metrics(empty_true, fp_pred)
    assert m2["dice"] == 0.0
    assert m2["precision"] == 0.0
    assert m2["specificity"] < 1.0

    # Case 3: Image with polyp, but model completely missed it
    has_polyp = np.zeros((64, 64), dtype=np.uint8)
    has_polyp[10:20, 10:20] = 1
    m3 = calculate_segmentation_metrics(has_polyp, empty_pred)
    assert m3["dice"] == 0.0
    assert m3["recall"] == 0.0
    assert m3["specificity"] == 1.0


def test_aggregation():
    sample1 = {"dice": 0.8, "iou": 0.66, "precision": 0.8, "recall": 0.8, "specificity": 0.95, "f1_score": 0.8, "accuracy": 0.9, "true_positives": 80, "false_positives": 20, "false_negatives": 20, "true_negatives": 880}
    sample2 = {"dice": 0.9, "iou": 0.81, "precision": 0.9, "recall": 0.9, "specificity": 0.98, "f1_score": 0.9, "accuracy": 0.95, "true_positives": 90, "false_positives": 10, "false_negatives": 10, "true_negatives": 890}
    agg = aggregate_dataset_metrics([sample1, sample2])
    assert agg["mean_dice"] == pytest.approx(0.85, abs=1e-4)
    assert agg["sample_count"] == 2
    assert "global_dice" in agg
