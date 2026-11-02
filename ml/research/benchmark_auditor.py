"""
Comprehensive Empirical Benchmark and Reliability Auditing Suite.
Evaluates:
- Baseline SegFormer, U-Net, LinkNet vs. Proposed MED-ECRA Consensus Fusion
- Clean test distribution vs. Optical Stress Distortions (Specular Glare, Motion Blur, Fluid Turbidity)
- Scale-stratified performance across Small (< 5%), Medium (5-20%), and Large (>= 20%) polyps
- Metrics: Dice, IoU, HD95, Boundary IoU, Precision, Recall, Specificity, CFRI, Latency.
Outputs: experiments/research_empirical_benchmark.json
"""

import os
import sys
import time
import json
import base64
import io
import numpy as np
from PIL import Image
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from ml.datasets.kvasir_dataset import resolve_dataset_path, get_deterministic_splits
from ml.inference.predictor import PolypPredictor
from ml.evaluation.metrics import calculate_segmentation_metrics
from ml.research.boundary_metrics import calculate_hd95, calculate_boundary_iou
from ml.research.stress_perturbations import (
    inject_specular_glare,
    inject_motion_blur,
    inject_fluid_turbidity
)
from ml.research.scale_stratification import get_polyp_scale_category
from ml.research.disagreement_fusion import run_med_ecra_analysis


def run_comprehensive_research_benchmark(
    max_test_samples: int = 50,  # Representative test sample for practical CPU computation
    seed: int = 42
) -> Dict[str, Any]:
    print("\n==========================================================================")
    print("  RUNNING RESEARCH AUDITING BENCHMARK (Kvasir-SEG Held-Out Test Split)")
    print(f"  Test Cohort: {max_test_samples} frames | Metrics: DSC, IoU, HD95, BIoU, CFRI")
    print("==========================================================================\n")

    dataset_path, report = resolve_dataset_path()
    train_p, val_p, test_p = get_deterministic_splits(report.valid_pairs, seed=seed)
    test_subset = test_p[:max_test_samples]

    predictor = PolypPredictor(device="cpu")
    models = ["segformer", "unet", "linknet"]

    # Pre-load models to ensure warm cache
    for m in models:
        predictor.get_model(m)

    # Buckets
    clean_results = {"segformer": [], "unet": [], "linknet": [], "proposed_consensus": []}
    glare_results = {"segformer": [], "unet": [], "linknet": [], "proposed_consensus": []}
    blur_results = {"segformer": [], "unet": [], "linknet": [], "proposed_consensus": []}
    turbid_results = {"segformer": [], "unet": [], "linknet": [], "proposed_consensus": []}

    scale_results = {
        "small": {"segformer": [], "unet": [], "linknet": [], "proposed_consensus": []},
        "medium": {"segformer": [], "unet": [], "linknet": [], "proposed_consensus": []},
        "large": {"segformer": [], "unet": [], "linknet": [], "proposed_consensus": []}
    }

    cfris = []
    latencies = {"segformer": [], "unet": [], "linknet": [], "proposed_consensus": []}

    start_bench = time.perf_counter()

    for idx, (img_path, mask_path) in enumerate(test_subset):
        with Image.open(img_path) as im, Image.open(mask_path) as ma:
            orig_pil = im.convert("RGB")
            gt_pil = ma.convert("L")
            orig_w, orig_h = orig_pil.size
            if gt_pil.size != (orig_w, orig_h):
                gt_pil = gt_pil.resize((orig_w, orig_h), resample=Image.NEAREST)
            gt_bin = (np.array(gt_pil) >= 127).astype(np.uint8)

        scale_cat = get_polyp_scale_category(gt_bin)
        if scale_cat == "negative":
            scale_cat = "small"

        # 1. Clean Inference
        model_preds = {}
        for m in models:
            t0 = time.perf_counter()
            pred = predictor.predict(img_path, model_name=m, threshold=0.5)
            dt = (time.perf_counter() - t0) * 1000.0
            latencies[m].append(dt)
            model_preds[m] = pred

            # Decode mask
            p_bytes = base64.b64decode(pred["mask_data_uri"].split(",", 1)[1])
            with Image.open(io.BytesIO(p_bytes)) as p_im:
                pred_bin = (np.array(p_im.convert("L")) >= 127).astype(np.uint8)

            m_scores = calculate_segmentation_metrics(gt_bin, pred_bin)
            m_scores["hd95"] = calculate_hd95(gt_bin, pred_bin)
            m_scores["biou"] = calculate_boundary_iou(gt_bin, pred_bin)
            clean_results[m].append(m_scores)
            scale_results[scale_cat][m].append(m_scores)

        # Proposed MED-ECRA Consensus Fusion
        t0 = time.perf_counter()
        fused = run_med_ecra_analysis(orig_pil, model_preds, threshold=0.50)
        dt_fused = (time.perf_counter() - t0) * 1000.0 + sum(latencies[m][-1] for m in models)
        latencies["proposed_consensus"].append(dt_fused)

        f_bytes = base64.b64decode(fused["fused_mask_data_uri"].split(",", 1)[1])
        with Image.open(io.BytesIO(f_bytes)) as f_im:
            fused_bin = (np.array(f_im.convert("L")) >= 127).astype(np.uint8)

        f_scores = calculate_segmentation_metrics(gt_bin, fused_bin)
        f_scores["hd95"] = calculate_hd95(gt_bin, fused_bin)
        f_scores["biou"] = calculate_boundary_iou(gt_bin, fused_bin)
        f_scores["cfri"] = fused["clinical_failure_risk_index"]
        cfris.append(fused["clinical_failure_risk_index"])

        clean_results["proposed_consensus"].append(f_scores)
        scale_results[scale_cat]["proposed_consensus"].append(f_scores)

        # 2. Stress Test: Specular Glare
        glare_img = inject_specular_glare(orig_pil, seed=idx)
        glare_preds = {}
        for m in models:
            pred = predictor.predict(glare_img, model_name=m, threshold=0.5)
            glare_preds[m] = pred
            p_bytes = base64.b64decode(pred["mask_data_uri"].split(",", 1)[1])
            with Image.open(io.BytesIO(p_bytes)) as p_im:
                pred_bin = (np.array(p_im.convert("L")) >= 127).astype(np.uint8)
            g_scores = calculate_segmentation_metrics(gt_bin, pred_bin)
            g_scores["hd95"] = calculate_hd95(gt_bin, pred_bin)
            glare_results[m].append(g_scores)

        fused_glare = run_med_ecra_analysis(Image.fromarray(glare_img), glare_preds, threshold=0.50)
        fg_bytes = base64.b64decode(fused_glare["fused_mask_data_uri"].split(",", 1)[1])
        with Image.open(io.BytesIO(fg_bytes)) as fg_im:
            fg_bin = (np.array(fg_im.convert("L")) >= 127).astype(np.uint8)
        fg_scores = calculate_segmentation_metrics(gt_bin, fg_bin)
        fg_scores["hd95"] = calculate_hd95(gt_bin, fg_bin)
        glare_results["proposed_consensus"].append(fg_scores)

        # 3. Stress Test: Motion Blur
        blur_img = inject_motion_blur(orig_pil, kernel_size=15, angle_degrees=30.0)
        blur_preds = {}
        for m in models:
            pred = predictor.predict(blur_img, model_name=m, threshold=0.5)
            blur_preds[m] = pred
            p_bytes = base64.b64decode(pred["mask_data_uri"].split(",", 1)[1])
            with Image.open(io.BytesIO(p_bytes)) as p_im:
                pred_bin = (np.array(p_im.convert("L")) >= 127).astype(np.uint8)
            b_scores = calculate_segmentation_metrics(gt_bin, pred_bin)
            b_scores["hd95"] = calculate_hd95(gt_bin, pred_bin)
            blur_results[m].append(b_scores)

        fused_blur = run_med_ecra_analysis(Image.fromarray(blur_img), blur_preds, threshold=0.50)
        fb_bytes = base64.b64decode(fused_blur["fused_mask_data_uri"].split(",", 1)[1])
        with Image.open(io.BytesIO(fb_bytes)) as fb_im:
            fb_bin = (np.array(fb_im.convert("L")) >= 127).astype(np.uint8)
        fb_scores = calculate_segmentation_metrics(gt_bin, fb_bin)
        fb_scores["hd95"] = calculate_hd95(gt_bin, fb_bin)
        blur_results["proposed_consensus"].append(fb_scores)

        # 4. Stress Test: Fluid Turbidity
        turbid_img = inject_fluid_turbidity(orig_pil)
        turbid_preds = {}
        for m in models:
            pred = predictor.predict(turbid_img, model_name=m, threshold=0.5)
            turbid_preds[m] = pred
            p_bytes = base64.b64decode(pred["mask_data_uri"].split(",", 1)[1])
            with Image.open(io.BytesIO(p_bytes)) as p_im:
                pred_bin = (np.array(p_im.convert("L")) >= 127).astype(np.uint8)
            tu_scores = calculate_segmentation_metrics(gt_bin, pred_bin)
            tu_scores["hd95"] = calculate_hd95(gt_bin, pred_bin)
            turbid_results[m].append(tu_scores)

        fused_turbid = run_med_ecra_analysis(Image.fromarray(turbid_img), turbid_preds, threshold=0.50)
        ft_bytes = base64.b64decode(fused_turbid["fused_mask_data_uri"].split(",", 1)[1])
        with Image.open(io.BytesIO(ft_bytes)) as ft_im:
            ft_bin = (np.array(ft_im.convert("L")) >= 127).astype(np.uint8)
        ft_scores = calculate_segmentation_metrics(gt_bin, ft_bin)
        ft_scores["hd95"] = calculate_hd95(gt_bin, ft_bin)
        turbid_results["proposed_consensus"].append(ft_scores)

        if (idx + 1) % 10 == 0 or (idx + 1) == max_test_samples:
            print(f"  Evaluated [{idx+1}/{max_test_samples}] frames | Fused Dice: {f_scores['dice']:.4f} | HD95: {f_scores['hd95']:.1f}px | CFRI: {fused['clinical_failure_risk_index']}%")

    total_bench_duration = round(time.perf_counter() - start_bench, 2)

    def agg(m_list):
        if not m_list:
            return {"dice": 0.0, "iou": 0.0, "hd95": 0.0, "biou": 0.0, "precision": 0.0, "recall": 0.0, "specificity": 0.0}
        return {
            "mean_dice": round(float(np.mean([x["dice"] for x in m_list])), 4),
            "std_dice": round(float(np.std([x["dice"] for x in m_list])), 4),
            "mean_iou": round(float(np.mean([x["iou"] for x in m_list])), 4),
            "mean_hd95": round(float(np.mean([x["hd95"] for x in m_list])), 2),
            "mean_biou": round(float(np.mean([x.get("biou", 0.0) for x in m_list])), 4),
            "mean_precision": round(float(np.mean([x["precision"] for x in m_list])), 4),
            "mean_recall": round(float(np.mean([x["recall"] for x in m_list])), 4),
            "mean_specificity": round(float(np.mean([x["specificity"] for x in m_list])), 4),
        }

    summary = {
        "benchmark_dataset": "Kvasir-SEG (Held-out Test Split)",
        "samples_evaluated": max_test_samples,
        "evaluation_duration_seconds": total_bench_duration,
        "clean_benchmarks": {k: agg(clean_results[k]) for k in clean_results},
        "stress_glare_benchmarks": {k: agg(glare_results[k]) for k in glare_results},
        "stress_blur_benchmarks": {k: agg(blur_results[k]) for k in blur_results},
        "stress_turbid_benchmarks": {k: agg(turbid_results[k]) for k in turbid_results},
        "scale_stratified_benchmarks": {
            scale: {k: agg(scale_results[scale][k]) for k in scale_results[scale]}
            for scale in ["small", "medium", "large"]
        },
        "latency_ms": {k: round(float(np.mean(latencies[k])), 1) for k in latencies},
        "mean_clinical_failure_risk_index": round(float(np.mean(cfris)), 1)
    }

    out_json = "experiments/research_empirical_benchmark.json"
    os.makedirs("experiments", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n==========================================================================")
    print(f"  BENCHMARK COMPLETED in {total_bench_duration}s ({total_bench_duration/60:.2f} mins)")
    print(f"  Saved to: {out_json}")
    print("==========================================================================\n")
    return summary


if __name__ == "__main__":
    run_comprehensive_research_benchmark(max_test_samples=30)

