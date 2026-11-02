"""
Held-Out Test Set Comprehensive Evaluator for 6 Segmentation & Fusion Methods.
Evaluates SegFormer-B0, U-Net, LinkNet, Majority Voting, Mean Probability Fusion, and ECCF Fusion
on the 100 untouched test frames from Kvasir-SEG (Seed=42).
Computes segmentation metrics, calibration metrics, uncertainty correlation,
failure detection AUROC/AUPRC, and statistical significance testing.
"""

import os
import sys
import time
import json
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from scipy import stats
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
    compute_epistemic_disagreement_field,
    fuse_predictions_evidence_calibrated
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


def compute_precision(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = 1e-7) -> float:
    p = (pred_bin > 0).astype(bool)
    g = (gt_bin > 0).astype(bool)
    tp = float(np.sum(p & g))
    fp = float(np.sum(p & (~g)))
    if tp + fp == 0:
        return 1.0
    return float(tp / (tp + fp + eps))


def evaluate_held_out_test_set(
    models_dict: Optional[Dict[str, nn.Module]] = None,
    device: Optional[str] = None,
    seed: int = 42,
    threshold: float = 0.50,
    sigma_w: float = 0.25,
    output_path: str = "experiments/full_test_evaluation_report.json"
) -> Dict[str, Any]:
    """
    Executes full comparative benchmarking on the untouched 100 test images.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_path, report = resolve_dataset_path()
    _, _, test_pairs = get_deterministic_splits(report.valid_pairs, seed=seed)
    n_test = len(test_pairs)

    print(f"\n=======================================================")
    print(f"EVALUATING ON UNTOUCHED HELD-OUT TEST SPLIT (N={n_test})")
    print(f"Hardware: {device.upper()} | Seed: {seed} | Threshold: {threshold}")
    print(f"=======================================================")

    # Load models if not provided
    if models_dict is None:
        models_dict = {}
        for m_name in ["segformer", "unet", "linknet"]:
            ckpt = f"ml/checkpoints/{m_name}_kvasir_trained.pt"
            if not os.path.isfile(ckpt):
                ckpt = f"ml/checkpoints/{m_name}_best.pt"
            models_dict[m_name] = create_model(m_name, device=device, checkpoint_path=ckpt, eval_mode=True, require_checkpoint=True)

    test_dataset = PolypSegmentationDataset(
        pairs=test_pairs,
        image_size=(256, 256),
        is_training=False,
        normalize_imagenet=True
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    methods = ["segformer", "unet", "linknet", "majority_voting", "mean_probability_fusion", "eccf_fusion"]

    # Storage for per-image results
    metrics_per_method = {m: {
        "dice": [], "iou": [], "sensitivity": [], "specificity": [], "precision": [], "recall": [],
        "brier_score": [], "expected_calibration_error": [], "latency_ms": []
    } for m in methods}

    uncertainty_results = {
        "jsd_pearson": [], "jsd_spearman": [], "jsd_auroc": [], "jsd_auprc": [],
        "entropy_pearson": [], "entropy_spearman": [], "entropy_auroc": [], "entropy_auprc": []
    }

    fusion_latencies = []
    total_pipeline_latencies = []

    print("Running inference across test cohort...")
    t_start = time.perf_counter()

    for idx, batch in enumerate(test_loader):
        img = batch["image"].to(device)
        gt_mask = batch["mask"][0, 0].cpu().numpy()
        gt_bin = (gt_mask >= 0.5).astype(np.uint8)

        t_pipe_start = time.perf_counter()
        model_probs = {}

        # 1. Individual model forward passes
        for m_name in ["segformer", "unet", "linknet"]:
            if torch.cuda.is_available() and device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = models_dict[m_name](img)
                probs = torch.sigmoid(logits)
            if torch.cuda.is_available() and device == "cuda":
                torch.cuda.synchronize()
            lat_ms = (time.perf_counter() - t0) * 1000.0

            prob_np = probs[0, 0].cpu().numpy().astype(np.float32)
            model_probs[m_name] = prob_np

            p_bin = (prob_np >= threshold).astype(np.uint8)
            metrics_per_method[m_name]["dice"].append(calculate_dice(p_bin, gt_bin))
            metrics_per_method[m_name]["iou"].append(calculate_iou(p_bin, gt_bin))
            metrics_per_method[m_name]["sensitivity"].append(calculate_sensitivity(p_bin, gt_bin))
            metrics_per_method[m_name]["specificity"].append(calculate_specificity(p_bin, gt_bin))
            metrics_per_method[m_name]["precision"].append(compute_precision(p_bin, gt_bin))
            metrics_per_method[m_name]["recall"].append(calculate_sensitivity(p_bin, gt_bin))
            metrics_per_method[m_name]["brier_score"].append(calculate_brier_score(prob_np, gt_bin))
            metrics_per_method[m_name]["expected_calibration_error"].append(calculate_expected_calibration_error(prob_np, gt_bin))
            metrics_per_method[m_name]["latency_ms"].append(lat_ms)

        # 2. Ensemble & Fusion Methods
        t_fus_start = time.perf_counter()
        probs_list = [model_probs["segformer"], model_probs["unet"], model_probs["linknet"]]
        mean_p = np.mean(np.stack(probs_list, axis=0), axis=0)
        votes = (probs_list[0] >= threshold).astype(int) + (probs_list[1] >= threshold).astype(int) + (probs_list[2] >= threshold).astype(int)
        maj_p = (votes / 3.0).astype(np.float32)
        maj_bin = (votes >= 2).astype(np.uint8)
        mean_bin = (mean_p >= threshold).astype(np.uint8)

        fused_p = fuse_predictions_evidence_calibrated(probs_list, sigma_w=sigma_w)
        fused_bin = (fused_p >= threshold).astype(np.uint8)

        fusion_lat = (time.perf_counter() - t_fus_start) * 1000.0
        fusion_latencies.append(fusion_lat)
        total_pipeline_latencies.append((time.perf_counter() - t_pipe_start) * 1000.0)

        # Majority voting metrics
        metrics_per_method["majority_voting"]["dice"].append(calculate_dice(maj_bin, gt_bin))
        metrics_per_method["majority_voting"]["iou"].append(calculate_iou(maj_bin, gt_bin))
        metrics_per_method["majority_voting"]["sensitivity"].append(calculate_sensitivity(maj_bin, gt_bin))
        metrics_per_method["majority_voting"]["specificity"].append(calculate_specificity(maj_bin, gt_bin))
        metrics_per_method["majority_voting"]["precision"].append(compute_precision(maj_bin, gt_bin))
        metrics_per_method["majority_voting"]["recall"].append(calculate_sensitivity(maj_bin, gt_bin))
        metrics_per_method["majority_voting"]["brier_score"].append(calculate_brier_score(maj_p, gt_bin))
        metrics_per_method["majority_voting"]["expected_calibration_error"].append(calculate_expected_calibration_error(maj_p, gt_bin))
        metrics_per_method["majority_voting"]["latency_ms"].append(fusion_lat)

        # Mean probability fusion metrics
        metrics_per_method["mean_probability_fusion"]["dice"].append(calculate_dice(mean_bin, gt_bin))
        metrics_per_method["mean_probability_fusion"]["iou"].append(calculate_iou(mean_bin, gt_bin))
        metrics_per_method["mean_probability_fusion"]["sensitivity"].append(calculate_sensitivity(mean_bin, gt_bin))
        metrics_per_method["mean_probability_fusion"]["specificity"].append(calculate_specificity(mean_bin, gt_bin))
        metrics_per_method["mean_probability_fusion"]["precision"].append(compute_precision(mean_bin, gt_bin))
        metrics_per_method["mean_probability_fusion"]["recall"].append(calculate_sensitivity(mean_bin, gt_bin))
        metrics_per_method["mean_probability_fusion"]["brier_score"].append(calculate_brier_score(mean_p, gt_bin))
        metrics_per_method["mean_probability_fusion"]["expected_calibration_error"].append(calculate_expected_calibration_error(mean_p, gt_bin))
        metrics_per_method["mean_probability_fusion"]["latency_ms"].append(fusion_lat)

        # ECCF fusion metrics
        metrics_per_method["eccf_fusion"]["dice"].append(calculate_dice(fused_bin, gt_bin))
        metrics_per_method["eccf_fusion"]["iou"].append(calculate_iou(fused_bin, gt_bin))
        metrics_per_method["eccf_fusion"]["sensitivity"].append(calculate_sensitivity(fused_bin, gt_bin))
        metrics_per_method["eccf_fusion"]["specificity"].append(calculate_specificity(fused_bin, gt_bin))
        metrics_per_method["eccf_fusion"]["precision"].append(compute_precision(fused_bin, gt_bin))
        metrics_per_method["eccf_fusion"]["recall"].append(calculate_sensitivity(fused_bin, gt_bin))
        metrics_per_method["eccf_fusion"]["brier_score"].append(calculate_brier_score(fused_p, gt_bin))
        metrics_per_method["eccf_fusion"]["expected_calibration_error"].append(calculate_expected_calibration_error(fused_p, gt_bin))
        metrics_per_method["eccf_fusion"]["latency_ms"].append(fusion_lat)

        # 3. Uncertainty evaluation
        jsd_map, _, h_pred = compute_epistemic_disagreement_field(probs_list, return_entropy=True)
        jsd_corr = calculate_uncertainty_error_correlation(jsd_map, mean_p, gt_bin)
        jsd_auc = calculate_failure_detection_auc(jsd_map, mean_bin, gt_bin)
        h_corr = calculate_uncertainty_error_correlation(h_pred, mean_p, gt_bin)
        h_auc = calculate_failure_detection_auc(h_pred, mean_bin, gt_bin)

        uncertainty_results["jsd_pearson"].append(jsd_corr["pearson_correlation"])
        uncertainty_results["jsd_spearman"].append(jsd_corr["spearman_correlation"])
        uncertainty_results["jsd_auroc"].append(jsd_auc["auroc"])
        uncertainty_results["jsd_auprc"].append(jsd_auc["auprc"])

        uncertainty_results["entropy_pearson"].append(h_corr["pearson_correlation"])
        uncertainty_results["entropy_spearman"].append(h_corr["spearman_correlation"])
        uncertainty_results["entropy_auroc"].append(h_auc["auroc"])
        uncertainty_results["entropy_auprc"].append(h_auc["auprc"])

    total_eval_time = round(time.perf_counter() - t_start, 2)
    print(f"Completed evaluation in {total_eval_time}s.")

    # 4. Statistical Summary (Mean, Std, 95% Confidence Intervals)
    summary_table = {}
    for m in methods:
        summary_table[m] = {}
        for metric_name, values in metrics_per_method[m].items():
            arr = np.array(values)
            m_val = float(np.mean(arr))
            s_val = float(np.std(arr))
            # 95% CI
            ci = 1.96 * s_val / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
            summary_table[m][metric_name] = {
                "mean": round(m_val, 4),
                "std": round(s_val, 4),
                "ci_95": [round(m_val - ci, 4), round(m_val + ci, 4)]
            }

    # 5. Statistical Significance vs ECCF (Wilcoxon Signed-Rank Test)
    eccf_dices = np.array(metrics_per_method["eccf_fusion"]["dice"])
    significance_vs_eccf = {}
    for m in ["segformer", "unet", "linknet", "majority_voting", "mean_probability_fusion"]:
        other_dices = np.array(metrics_per_method[m]["dice"])
        try:
            stat, p_val = stats.wilcoxon(eccf_dices, other_dices)
            significance_vs_eccf[m] = {
                "wilcoxon_stat": float(round(stat, 2)),
                "p_value": float(round(p_val, 6)),
                "statistically_significant_p05": bool(p_val < 0.05)
            }
        except Exception:
            significance_vs_eccf[m] = {"wilcoxon_stat": 0.0, "p_value": 1.0, "statistically_significant_p05": False}

    # Uncertainty Summary
    uncertainty_summary = {
        k: {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4)
        } for k, vals in uncertainty_results.items()
    }

    report_payload = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "Kvasir-SEG (Held-Out Test Split)",
        "test_samples_count": n_test,
        "seed": seed,
        "device": device,
        "total_evaluation_seconds": total_eval_time,
        "average_pipeline_latency_ms": round(float(np.mean(total_pipeline_latencies)), 2),
        "average_fusion_latency_ms": round(float(np.mean(fusion_latencies)), 2),
        "summary_metrics": summary_table,
        "statistical_tests": significance_vs_eccf,
        "uncertainty_evaluation": uncertainty_summary
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report_payload, f, indent=2)

    print(f"Saved Full Evaluation Report to: {output_path}")
    return report_payload


if __name__ == "__main__":
    evaluate_held_out_test_set()
