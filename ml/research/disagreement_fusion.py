"""
Tri-Model Disagreement-Guided Polyp Segmentation Reliability Assessment.

Mathematical Formulation:
1. Tri-Architectural Epistemic Disagreement Field (JSD across continuous probabilities of SegFormer, U-Net, LinkNet).
2. Predictive Entropy Field H(mean_p).
3. Boundary Disagreement Field (Perimeter-specific ambiguity with empty-consensus fallback).
4. Evidence-Calibrated Consensus Fusion (ECCF) with configurable manifold distance parameter sigma_w.
5. Non-Clinical Model Disagreement Indicator (MDI) evaluating algorithmic concordance without clinical diagnosis claims.
"""

import os
import io
import time
import base64
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional, Union


def compute_bernoulli_entropy(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Computes binary Shannon entropy H(p) = -p log2(p) - (1-p) log2(1-p).
    Numerically stabilized with clamping in [eps, 1 - eps].
    Max entropy is 1.0 at p = 0.5.
    """
    p_clamped = np.clip(p, eps, 1.0 - eps)
    entropy = -(p_clamped * np.log2(p_clamped) + (1.0 - p_clamped) * np.log2(1.0 - p_clamped))
    return np.clip(entropy, 0.0, 1.0).astype(np.float32)


def compute_epistemic_disagreement_field(
    probs: List[np.ndarray],
    eps: float = 1e-7,
    return_entropy: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Computes Jensen-Shannon Divergence across K continuous probability maps:
      mean_p = (1/K) * sum(p_k)
      H_pred = H(mean_p)               # Predictive entropy
      mean_H = (1/K) * sum(H(p_k))     # Mean individual entropy
      JSD = H_pred - mean_H             # Epistemic disagreement field

    Returns:
      If return_entropy is False (default for backwards compatibility):
        jsd: (H, W) array in [0, 1] representing epistemic disagreement
        mean_p: (H, W) consensus probability map in [0, 1]
      If return_entropy is True:
        jsd, mean_p, h_pred
    """
    probs_stack = np.stack(probs, axis=0).astype(np.float32)  # (K, H, W)
    mean_p = np.mean(probs_stack, axis=0)  # (H, W)

    h_pred = compute_bernoulli_entropy(mean_p, eps=eps)
    h_individuals = [compute_bernoulli_entropy(p, eps=eps) for p in probs]
    mean_h_indiv = np.mean(np.stack(h_individuals, axis=0), axis=0)

    jsd = np.clip(h_pred - mean_h_indiv, 0.0, 1.0).astype(np.float32)
    if return_entropy:
        return jsd, mean_p.astype(np.float32), h_pred.astype(np.float32)
    return jsd, mean_p.astype(np.float32)


def compute_boundary_ambiguity_field(
    mean_p: np.ndarray,
    jsd: np.ndarray,
    probs: Optional[List[np.ndarray]] = None,
    dilation_radius: int = 3
) -> Tuple[np.ndarray, float]:
    """
    Isolates the lesion perimeter band and calculates boundary-specific disagreement.
    Handles empty consensus without falsely reporting zero uncertainty when individual
    models express conflicting localized activations.

    Returns:
      baf: (H, W) boundary ambiguity map
      bdr: scalar Boundary Disagreement Ratio in [0, 1]
    """
    bin_consensus = (mean_p >= 0.50).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius * 2 + 1, dilation_radius * 2 + 1))

    if np.any(bin_consensus > 0):
        boundary_band = cv2.morphologyEx(bin_consensus, cv2.MORPH_GRADIENT, kernel) > 0
    else:
        # Empty consensus handling: Check if individual models differ significantly
        if probs is not None and len(probs) > 1:
            probs_stack = np.stack(probs, axis=0)
            model_spread = np.max(probs_stack, axis=0) - np.min(probs_stack, axis=0)
        else:
            model_spread = jsd

        # When models conflict or have partial activations, extract candidate region perimeter
        has_variance = np.any(model_spread >= 0.15) or np.any(mean_p >= 0.15)
        if has_variance:
            soft_thresh = max(0.15, float(np.percentile(mean_p[mean_p > 0.05], 75))) if np.any(mean_p > 0.05) else 0.15
            soft_bin = (mean_p >= soft_thresh).astype(np.uint8)
            boundary_band = cv2.morphologyEx(soft_bin, cv2.MORPH_GRADIENT, kernel) > 0
            if not np.any(boundary_band):
                boundary_band = model_spread >= 0.15
        else:
            boundary_band = np.zeros_like(mean_p, dtype=bool)

    baf = jsd * boundary_band.astype(np.float32)

    band_pixels = int(np.sum(boundary_band))
    if band_pixels == 0:
        mean_jsd = float(np.mean(jsd))
        if mean_jsd >= 0.02:
            # Models disagree globally or locally without forming a discrete perimeter
            bdr = mean_jsd
        else:
            bdr = 0.0
    else:
        bdr = float(np.sum(baf) / band_pixels)

    return baf.astype(np.float32), round(bdr, 4)


def fuse_predictions_evidence_calibrated(
    probs: List[np.ndarray],
    sigma_w: float = 0.25
) -> np.ndarray:
    """
    Evidence-Calibrated Consensus Fusion (ECCF).
    Weights each model's prediction exponentially by its closeness to the consensus manifold:
      w_k(x, y) = exp( - (p_k - mean_p)^2 / (2 * sigma_w^2) )
      P_fused = sum(w_k * p_k) / sum(w_k)
    """
    if sigma_w <= 0.0:
        raise ValueError(f"sigma_w must be strictly positive (> 0.0), got {sigma_w}")

    probs_stack = np.stack(probs, axis=0).astype(np.float32)  # (K, H, W)
    mean_p = np.mean(probs_stack, axis=0)  # (H, W)

    # Compute spatial weights
    weights = np.exp(-((probs_stack - mean_p) ** 2) / (2.0 * (sigma_w ** 2)))  # (K, H, W)
    weights_sum = np.sum(weights, axis=0) + 1e-7

    fused_p = np.sum(weights * probs_stack, axis=0) / weights_sum
    return np.clip(fused_p, 0.0, 1.0).astype(np.float32)


def calculate_model_disagreement_indicator(
    jsd: np.ndarray,
    bdr: float,
    probs: List[np.ndarray],
    alpha: float = 0.40,
    beta: float = 0.35,
    gamma: float = 0.25
) -> Tuple[float, str]:
    """
    Computes a normalized single-frame Model Disagreement Indicator (MDI) in [0, 100]%.
    Non-clinical heuristic evaluating multi-architecture algorithmic divergence:
      1. Global mean epistemic disagreement (JSD)
      2. Boundary disagreement ratio (BDR)
      3. Multi-model pairwise Jaccard discordance (1 - Mean Pairwise IoU)
    """
    mean_jsd = float(np.mean(jsd))

    # Compute pairwise Jaccard discordance
    bin_masks = [(p >= 0.5).astype(bool) for p in probs]
    k = len(bin_masks)
    jaccards = []
    for i in range(k):
        for j in range(i + 1, k):
            inter = np.sum(bin_masks[i] & bin_masks[j])
            union = np.sum(bin_masks[i] | bin_masks[j])
            if union == 0:
                jaccards.append(1.0)
            else:
                jaccards.append(inter / union)

    mean_pairwise_iou = float(np.mean(jaccards)) if jaccards else 1.0
    discordance = 1.0 - mean_pairwise_iou

    raw_score = (alpha * mean_jsd * 4.0) + (beta * bdr * 2.5) + (gamma * discordance)
    score_pct = float(round(np.clip(raw_score * 100.0, 0.0, 100.0), 1))

    if score_pct < 20.0:
        level = "High Consensus (Low Inter-Model Disagreement)"
    elif score_pct < 45.0:
        level = "Moderate Disagreement (Perimeter Variance)"
    else:
        level = "Elevated Disagreement (Multi-Model Variance Flagged)"

    return score_pct, level


# Backwards compatibility alias
def calculate_clinical_failure_risk_index(
    jsd: np.ndarray,
    bdr: float,
    probs: List[np.ndarray],
    alpha: float = 0.40,
    beta: float = 0.35,
    gamma: float = 0.25
) -> Tuple[float, str]:
    """Alias for calculate_model_disagreement_indicator for backward compatibility."""
    return calculate_model_disagreement_indicator(jsd, bdr, probs, alpha=alpha, beta=beta, gamma=gamma)


def generate_heatmap_overlay(
    heat_map: np.ndarray,
    orig_np: np.ndarray,
    alpha: float = 0.60,
    colormap: int = cv2.COLORMAP_INFERNO
) -> str:
    """Renders a base64 Data URI of a colormapped heat field alpha-blended over the endoscopy frame."""
    norm = np.clip(heat_map * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, colormap)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    blended = (orig_np.astype(np.float32) * (1.0 - alpha) + colored_rgb.astype(np.float32) * alpha).astype(np.uint8)
    pil_img = Image.fromarray(blended)

    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")


def array_to_data_uri(arr: np.ndarray, mode: str = "L", format: str = "PNG") -> str:
    """Encodes a numpy array (uint8 or [0, 1] float) into a base64 data URI."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


def run_tri_model_disagreement_analysis(
    orig_pil: Image.Image,
    model_prob_maps: Dict[str, np.ndarray],
    model_predictions: Optional[Dict[str, Dict[str, Any]]] = None,
    threshold: float = 0.50,
    sigma_w: float = 0.25
) -> Dict[str, Any]:
    """
    Executes Tri-Model Disagreement-Guided Reliability Assessment given continuous
    probability maps from SegFormer, U-Net, and LinkNet.
    """
    orig_w, orig_h = orig_pil.size
    orig_np = np.array(orig_pil.convert("RGB"))

    model_keys = list(model_prob_maps.keys())
    probs = []
    for k in model_keys:
        p = model_prob_maps[k]
        if p.shape != (orig_h, orig_w):
            p = cv2.resize(p, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        probs.append(np.clip(p, 0.0, 1.0).astype(np.float32))

    # 1. Epistemic Disagreement & Predictive Entropy Fields
    jsd, mean_p, h_pred = compute_epistemic_disagreement_field(probs, return_entropy=True)

    # 2. Boundary Disagreement Field
    baf, bdr = compute_boundary_ambiguity_field(mean_p, jsd, probs=probs)

    # 3. Evidence-Calibrated Consensus Fusion (ECCF)
    fused_p = fuse_predictions_evidence_calibrated(probs, sigma_w=sigma_w)
    fused_bin = (fused_p >= threshold).astype(np.uint8) * 255
    consensus_bin = (mean_p >= threshold).astype(np.uint8) * 255

    # 4. Model Disagreement Indicator
    mdi_pct, reliability_level = calculate_model_disagreement_indicator(jsd, bdr, probs)

    # 5. Overlays and Heatmap Data URIs
    jsd_heatmap_uri = generate_heatmap_overlay(jsd, orig_np, alpha=0.55, colormap=cv2.COLORMAP_INFERNO)
    h_pred_heatmap_uri = generate_heatmap_overlay(h_pred, orig_np, alpha=0.55, colormap=cv2.COLORMAP_MAGMA)
    boundary_uri = generate_heatmap_overlay(baf, orig_np, alpha=0.65, colormap=cv2.COLORMAP_JET)

    mean_prob_uri = array_to_data_uri(mean_p, mode="L")
    fused_prob_uri = array_to_data_uri(fused_p, mode="L")
    consensus_mask_uri = array_to_data_uri(consensus_bin, mode="L")
    fused_mask_uri = array_to_data_uri(fused_bin, mode="L")

    # Consensus overlay (Turquoise)
    consensus_overlay_np = orig_np.copy()
    if np.any(consensus_bin > 0):
        mask_bool = consensus_bin > 0
        turq = (0, 220, 180)
        for c in range(3):
            consensus_overlay_np[:, :, c] = np.where(
                mask_bool,
                (consensus_overlay_np[:, :, c] * 0.55 + turq[c] * 0.45).astype(np.uint8),
                consensus_overlay_np[:, :, c]
            )
        contours, _ = cv2.findContours(consensus_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(consensus_overlay_np, contours, -1, (255, 255, 255), 2)
        cv2.drawContours(consensus_overlay_np, contours, -1, turq, 1)

    consensus_overlay_uri = array_to_data_uri(consensus_overlay_np, mode="RGB")

    # Fused overlay (Vivid Emerald)
    fused_overlay_np = orig_np.copy()
    has_polyp = int(np.sum(fused_bin > 0)) > 25
    if has_polyp:
        mask_bool = fused_bin > 0
        emerald = (0, 230, 160)
        for c in range(3):
            fused_overlay_np[:, :, c] = np.where(
                mask_bool,
                (fused_overlay_np[:, :, c] * 0.55 + emerald[c] * 0.45).astype(np.uint8),
                fused_overlay_np[:, :, c]
            )
        contours, _ = cv2.findContours(fused_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(fused_overlay_np, contours, -1, (255, 255, 255), 2)
        cv2.drawContours(fused_overlay_np, contours, -1, emerald, 1)

    fused_overlay_uri = array_to_data_uri(fused_overlay_np, mode="RGB")
    polyp_pixels = int(np.sum(fused_bin > 0))
    area_pct = round((polyp_pixels / (orig_w * orig_h)) * 100.0, 2)

    return {
        "models_evaluated": model_keys,
        "polyp_detected": bool(has_polyp),
        "polyp_area_percentage": area_pct,
        "mean_probability_map_data_uri": mean_prob_uri,
        "fused_probability_map_data_uri": fused_prob_uri,
        "consensus_mask_data_uri": consensus_mask_uri,
        "consensus_overlay_data_uri": consensus_overlay_uri,
        "fused_mask_data_uri": fused_mask_uri,
        "fused_overlay_data_uri": fused_overlay_uri,
        "jsd_heatmap_data_uri": jsd_heatmap_uri,
        "predictive_entropy_heatmap_data_uri": h_pred_heatmap_uri,
        "boundary_disagreement_heatmap_data_uri": boundary_uri,
        "model_disagreement_indicator": mdi_pct,
        "reliability_level": reliability_level,
        "mean_jsd": float(round(float(np.mean(jsd)), 4)),
        "mean_predictive_entropy": float(round(float(np.mean(h_pred)), 4)),
        "boundary_disagreement_ratio": float(round(bdr, 4)),
        # Backwards compatibility fields for MED-ECRA
        "clinical_failure_risk_index": mdi_pct,
        "risk_level": reliability_level,
        "disagreement_heatmap_data_uri": jsd_heatmap_uri,
        "boundary_ambiguity_data_uri": boundary_uri,
        "mean_epistemic_disagreement": float(round(float(np.mean(jsd)), 4)),
        "models_fused": model_keys
    }


def run_med_ecra_analysis(
    orig_pil: Image.Image,
    model_predictions: Dict[str, Dict[str, Any]],
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Executes full MED-ECRA analysis given outputs from individual models.
    Supports continuous probability maps if present; otherwise decodes masks.
    """
    orig_w, orig_h = orig_pil.size
    model_prob_maps = {}

    for k, v in model_predictions.items():
        if "probability_map_data_uri" in v and v["probability_map_data_uri"]:
            uri = v["probability_map_data_uri"]
            raw_b64 = uri.split(",", 1)[1] if "," in uri else uri
            p_bytes = base64.b64decode(raw_b64)
            with Image.open(io.BytesIO(p_bytes)) as p_img:
                m_arr = np.array(p_img.convert("L")).astype(np.float32) / 255.0
        elif "mask_data_uri" in v and v["mask_data_uri"]:
            uri = v["mask_data_uri"]
            raw_b64 = uri.split(",", 1)[1] if "," in uri else uri
            m_bytes = base64.b64decode(raw_b64)
            with Image.open(io.BytesIO(m_bytes)) as m_img:
                m_arr = np.array(m_img.convert("L")).astype(np.float32) / 255.0
        else:
            m_arr = np.zeros((orig_h, orig_w), dtype=np.float32)

        if m_arr.shape != (orig_h, orig_w):
            m_arr = cv2.resize(m_arr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        model_prob_maps[k] = m_arr

    res = run_tri_model_disagreement_analysis(
        orig_pil=orig_pil,
        model_prob_maps=model_prob_maps,
        model_predictions=model_predictions,
        threshold=threshold,
        sigma_w=0.25
    )
    return res
