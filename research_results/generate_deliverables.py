"""
Generate Publication Deliverables: CSVs, Figures, and Metric Summary Tables.
"""

import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
output_dir = "research_results"
os.makedirs(output_dir, exist_ok=True)

# 1. Load Experimental Reports
with open("experiments/full_test_evaluation_report.json") as f:
    eval_data = json.load(f)

with open("experiments/ablation_study_report.json") as f:
    ablation_data = json.load(f)

with open("experiments/kvasir_full_training_all_models.json") as f:
    train_data = json.load(f)

with open("ml/checkpoints/temperature_scaling_params.json") as f:
    temp_data = json.load(f)

# -------------------------------------------------------------
# 2. Export CSV 1: 6-Method Benchmark Comparison Matrix
# -------------------------------------------------------------
csv_benchmark_path = os.path.join(output_dir, "benchmark_matrix.csv")
with open(csv_benchmark_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Method", "Architecture Family", "Dice Mean", "Dice Std", "Dice 95% CI Lower", "Dice 95% CI Upper",
        "IoU Mean", "Sensitivity (Recall)", "Specificity", "Precision", "Brier Score", "ECE",
        "Pipeline Latency (ms)", "Wilcoxon vs ECCF (p-value)", "Statistically Significant (p < 0.05)"
    ])
    
    families = {
        "segformer": "Transformer (Mix Transformer)",
        "unet": "CNN (Concatenative Skip)",
        "linknet": "CNN (Additive Residual)",
        "majority_voting": "Ensemble (Discrete Voting)",
        "mean_probability_fusion": "Ensemble (Linear Average)",
        "eccf_fusion": "Consensus (Evidence-Calibrated Exponential)"
    }
    
    for m, metrics in eval_data["summary_metrics"].items():
        sig = eval_data["statistical_tests"].get(m, {})
        writer.writerow([
            m,
            families.get(m, "Ensemble"),
            f"{metrics['dice']['mean']:.4f}",
            f"{metrics['dice']['std']:.4f}",
            f"{metrics['dice']['ci_95'][0]:.4f}",
            f"{metrics['dice']['ci_95'][1]:.4f}",
            f"{metrics['iou']['mean']:.4f}",
            f"{metrics['sensitivity']['mean']:.4f}",
            f"{metrics['specificity']['mean']:.4f}",
            f"{metrics['precision']['mean']:.4f}",
            f"{metrics['brier_score']['mean']:.4f}",
            f"{metrics['expected_calibration_error']['mean']:.4f}",
            f"{metrics['latency_ms']['mean']:.2f}",
            sig.get("p_value", "Baseline"),
            sig.get("statistically_significant_p05", "N/A")
        ])

print(f"Exported: {csv_benchmark_path}")

# -------------------------------------------------------------
# 3. Export CSV 2: Component Ablation Study Matrix
# -------------------------------------------------------------
csv_ablation_path = os.path.join(output_dir, "ablation_matrix.csv")
with open(csv_ablation_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Configuration", "Component Description", "Dice Mean", "Dice Std", "IoU Mean", "AUROC", "Pearson r", "Spearman r"
    ])
    for c_name, c_data in ablation_data["component_ablation_results"].items():
        writer.writerow([
            c_name,
            c_name.replace("_", " ").title(),
            f"{c_data['dice']['mean']:.4f}" if "dice" in c_data else "N/A",
            f"{c_data['dice']['std']:.4f}" if "dice" in c_data else "N/A",
            f"{c_data['iou']['mean']:.4f}" if "iou" in c_data else "N/A",
            f"{c_data['auroc']['mean']:.4f}" if "auroc" in c_data else "N/A",
            f"{c_data['pearson']['mean']:.4f}" if "pearson" in c_data else "N/A",
            f"{c_data['spearman']['mean']:.4f}" if "spearman" in c_data else "N/A",
        ])

print(f"Exported: {csv_ablation_path}")

# -------------------------------------------------------------
# 4. Generate Plot 1: 6-Method Benchmark Comparison Bar Chart
# -------------------------------------------------------------
methods = list(eval_data["summary_metrics"].keys())
display_names = ["SegFormer-B0", "U-Net", "LinkNet", "Majority Vote", "Mean Fusion", "ECCF (Proposed)"]
dices = [eval_data["summary_metrics"][m]["dice"]["mean"] * 100 for m in methods]
dice_errs = [eval_data["summary_metrics"][m]["dice"]["std"] * 100 / np.sqrt(100) * 1.96 for m in methods]
ious = [eval_data["summary_metrics"][m]["iou"]["mean"] * 100 for m in methods]
specificities = [eval_data["summary_metrics"][m]["specificity"]["mean"] * 100 for m in methods]

fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
x = np.arange(len(methods))
width = 0.28

rects1 = ax1.bar(x - width, dices, width, yerr=dice_errs, capsize=4, label='Dice Score (%) ± 95% CI', color='#8b5cf6', alpha=0.9)
rects2 = ax1.bar(x, ious, width, label='IoU (Jaccard %) ', color='#06b6d4', alpha=0.9)
rects3 = ax1.bar(x + width, specificities, width, label='Specificity (%)', color='#10b981', alpha=0.8)

ax1.set_ylabel('Metric Value (%)', fontsize=11, fontweight='bold')
ax1.set_title('Comparative Benchmark Across 6 Methods on Held-Out Test Cohort (N=100, Kvasir-SEG)', fontsize=12, fontweight='bold', pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(display_names, rotation=15, ha='right', fontsize=10, fontweight='semibold')
ax1.set_ylim(0, 100)
ax1.legend(loc='lower right', frameon=True)
plt.tight_layout()
benchmark_img_path = os.path.join(output_dir, "benchmark_comparison.png")
plt.savefig(benchmark_img_path)
plt.close()
print(f"Generated: {benchmark_img_path}")

# -------------------------------------------------------------
# 5. Generate Plot 2: Sigma Sensitivity Curve
# -------------------------------------------------------------
sigmas = []
sigma_dices = []
sigma_eces = []
sweep = ablation_data["sigma_sensitivity_validation"]["sweep_results"]
for k, v in sweep.items():
    sigmas.append(v["sigma"])
    sigma_dices.append(v["mean_dice"] * 100)
    sigma_eces.append(v["expected_calibration_error"])

fig, ax_dice = plt.subplots(figsize=(8, 4.5), dpi=300)
color_dice = '#7c3aed'
ax_dice.set_xlabel('ECCF Exponential Bandwidth Parameter (σ_w)', fontsize=11, fontweight='bold')
ax_dice.set_ylabel('Validation Mean Dice Score (%)', color=color_dice, fontsize=11, fontweight='bold')
ax_dice.plot(sigmas, sigma_dices, color=color_dice, marker='o', linewidth=2.5, markersize=7, label='Mean Dice (%)')
ax_dice.tick_params(axis='y', labelcolor=color_dice)
ax_dice.set_ylim(43.0, 44.5)

ax_ece = ax_dice.twinx()
color_ece = '#f59e0b'
ax_ece.set_ylabel('Expected Calibration Error (ECE)', color=color_ece, fontsize=11, fontweight='bold')
ax_ece.plot(sigmas, sigma_eces, color=color_ece, marker='s', linestyle='--', linewidth=2, markersize=6, label='ECE')
ax_ece.tick_params(axis='y', labelcolor=color_ece)

plt.title('Validation-Set Bandwidth Sensitivity Analysis (σ_w ∈ [0.10, 0.50])', fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
ablation_img_path = os.path.join(output_dir, "ablation_sensitivity.png")
plt.savefig(ablation_img_path)
plt.close()
print(f"Generated: {ablation_img_path}")

# -------------------------------------------------------------
# 6. Generate Plot 3: Calibration & Uncertainty Quality
# -------------------------------------------------------------
fig, (ax_cal, ax_auc) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

models_cal = ["SegFormer", "U-Net", "LinkNet", "Mean Fusion", "ECCF Fusion"]
eces = [
    eval_data["summary_metrics"]["segformer"]["expected_calibration_error"]["mean"],
    eval_data["summary_metrics"]["unet"]["expected_calibration_error"]["mean"],
    eval_data["summary_metrics"]["linknet"]["expected_calibration_error"]["mean"],
    eval_data["summary_metrics"]["mean_probability_fusion"]["expected_calibration_error"]["mean"],
    eval_data["summary_metrics"]["eccf_fusion"]["expected_calibration_error"]["mean"]
]
briers = [
    eval_data["summary_metrics"]["segformer"]["brier_score"]["mean"],
    eval_data["summary_metrics"]["unet"]["brier_score"]["mean"],
    eval_data["summary_metrics"]["linknet"]["brier_score"]["mean"],
    eval_data["summary_metrics"]["mean_probability_fusion"]["brier_score"]["mean"],
    eval_data["summary_metrics"]["eccf_fusion"]["brier_score"]["mean"]
]

x_m = np.arange(len(models_cal))
w_m = 0.35
ax_cal.bar(x_m - w_m/2, eces, w_m, label='Expected Calibration Error (ECE)', color='#f43f5e', alpha=0.85)
ax_cal.bar(x_m + w_m/2, briers, w_m, label='Brier Score (MSE)', color='#3b82f6', alpha=0.85)
ax_cal.set_ylabel('Error Metric (Lower is Better)', fontsize=10, fontweight='bold')
ax_cal.set_title('Calibration & Probability Reliability on Test Set', fontsize=11, fontweight='bold')
ax_cal.set_xticks(x_m)
ax_cal.set_xticklabels(models_cal, rotation=20, ha='right', fontsize=9)
ax_cal.legend(frameon=True, fontsize=9)

# Uncertainty Failure Detection AUROC comparison
forms = ["1. JSD Epistemic", "2. Shannon Entropy", "3. Model Variance", "4. Boundary Ambiguity", "5. Combined (Proposed)"]
aucs = [0.5011, 0.6815, 0.5420, 0.5890, 0.6608]
colors = ['#94a3b8', '#10b981', '#94a3b8', '#38bdf8', '#8b5cf6']
ax_auc.barh(forms, aucs, color=colors, height=0.55, alpha=0.9)
ax_auc.axvline(0.5, color='gray', linestyle=':', label='Random Guessing (0.50)')
ax_auc.set_xlabel('Failure Detection AUROC (Higher is Better)', fontsize=10, fontweight='bold')
ax_auc.set_title('Uncertainty Formulation Failure Prediction AUROC', fontsize=11, fontweight='bold')
ax_auc.set_xlim(0.4, 0.75)
ax_auc.legend(loc='lower right', frameon=True, fontsize=9)

plt.tight_layout()
calibration_img_path = os.path.join(output_dir, "calibration_reliability.png")
plt.savefig(calibration_img_path)
plt.close()
print(f"Generated: {calibration_img_path}")
print("All deliverables generated successfully.")

