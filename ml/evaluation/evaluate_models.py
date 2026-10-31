"""
Empirical Model Evaluation and Comparative Benchmarking Suite.
Executes U-Net, LinkNet, and SegFormer on identical test endoscopy frames with genuine ground-truth masks.
Calculates strict Dice (DSC), IoU (Jaccard), Precision, Recall, Specificity, and Latency.
"""

import os
import sys
import json
import time
import base64
import io
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from ml.inference.predictor import PolypPredictor
from ml.evaluation.metrics import calculate_segmentation_metrics, aggregate_dataset_metrics
from ml.models.registry import MODEL_CATALOG, get_available_models_metadata


def run_comparative_evaluation():
    print("=========================================================================================")
    print("  EMPIRICAL DEEP POLYP SEGMENTATION COMPARATIVE BENCHMARK")
    print("  Models: U-Net | LinkNet | SegFormer")
    print("  Ground-Truth Evaluation: Dice (F1), IoU (Jaccard), Precision, Recall, Specificity")
    print("=========================================================================================\n")

    predictor = PolypPredictor(device="cpu")
    models = ["segformer", "unet", "linknet"]

    # Identical test images with verified ground truth masks
    test_cases = [
        {
            "id": "sessile",
            "name": "Case 1: Sessile Polyp (Cecum)",
            "image": "data/sample_images/sample_polyp_sessile.jpg",
            "mask": "data/sample_images/sample_polyp_sessile_mask.png",
            "has_polyp": True
        },
        {
            "id": "elevated",
            "name": "Case 2: Elevated Polyp (Descending Colon)",
            "image": "data/sample_images/sample_polyp_elevated.jpg",
            "mask": "data/sample_images/sample_polyp_elevated_mask.png",
            "has_polyp": True
        },
        {
            "id": "normal",
            "name": "Case 3: Healthy Mucosa Control",
            "image": "data/sample_images/sample_normal_mucosa.jpg",
            "mask": "data/sample_images/sample_normal_mucosa_mask.png",
            "has_polyp": False
        }
    ]

    all_results = {}
    model_summaries = {}

    for model_name in models:
        print(f">>> Evaluating Model: {model_name.upper()} <<<")
        model_results = []
        latencies = []

        for case in test_cases:
            img_path = case["image"]
            mask_path = case["mask"]

            # Load ground truth binary mask
            with Image.open(mask_path) as m_img:
                gt_mask = (np.array(m_img.convert("L")) >= 127).astype(np.uint8)

            # Predict using model
            start = time.perf_counter()
            pred = predictor.predict(img_path, model_name=model_name, threshold=0.5)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            # Decode predicted binary mask
            mask_b64 = pred["mask_data_uri"].split(",", 1)[1]
            pred_mask_bytes = base64.b64decode(mask_b64)
            with Image.open(io.BytesIO(pred_mask_bytes)) as p_img:
                pred_mask = (np.array(p_img.convert("L")) >= 127).astype(np.uint8)

            # Calculate mathematical metrics
            m = calculate_segmentation_metrics(gt_mask, pred_mask, threshold=0.5)
            m["latency_ms"] = round(pred["inference_time_ms"], 2)
            m["case_id"] = case["id"]
            m["case_name"] = case["name"]
            m["polyp_detected"] = pred["polyp_detected"]
            m["polyp_area_pct"] = pred["polyp_area_percentage"]

            model_results.append(m)
            latencies.append(pred["inference_time_ms"])

            print(f"  • {case['id']:<10} | Latency: {pred['inference_time_ms']:5.1f}ms | Dice: {m['dice']:.4f} | IoU: {m['iou']:.4f} | Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f} | Spec: {m['specificity']:.4f}")

        # Compute aggregate metrics across test cases
        agg = aggregate_dataset_metrics(model_results)
        agg["mean_latency_ms"] = round(float(np.mean(latencies)), 2)
        model_summaries[model_name] = {
            "per_case": model_results,
            "aggregate": agg
        }
        all_results[model_name] = model_summaries[model_name]
        print(f"  --> MEAN {model_name.upper()}: Dice={agg['mean_dice']:.4f} | IoU={agg['mean_iou']:.4f} | Latency={agg['mean_latency_ms']}ms\n")

    # Save to experiments directory
    os.makedirs("experiments", exist_ok=True)
    out_path = "experiments/evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("=========================================================================================")
    print(f"  SUMMARY COMPARATIVE TABLE (Saved to {out_path})")
    print("=========================================================================================")
    print(f"{'Architecture':<12} | {'Mean Dice':<10} | {'Mean IoU':<10} | {'Precision':<10} | {'Recall':<10} | {'Specificity':<11} | {'Latency':<8}")
    print("-" * 85)
    for model_name in models:
        agg = model_summaries[model_name]["aggregate"]
        print(f"{model_name.upper():<12} | {agg['mean_dice']:<10.4f} | {agg['mean_iou']:<10.4f} | {agg['mean_precision']:<10.4f} | {agg['mean_recall']:<10.4f} | {agg['mean_specificity']:<11.4f} | {agg['mean_latency_ms']:<5.1f} ms")
    print("=========================================================================================\n")

    return all_results


if __name__ == "__main__":
    run_comparative_evaluation()
