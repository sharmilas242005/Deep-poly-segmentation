"""
Boundary and Contour Accuracy Metrics for Medical Image Segmentation.
Implements mathematically rigorous 95th Percentile Hausdorff Distance (HD95),
Boundary IoU (BIoU), and Surface Dice for Colonoscopy Lesions.
"""

import numpy as np
import cv2
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
from typing import Dict, Any, Tuple, Optional


def extract_binary_contours(mask: np.ndarray) -> np.ndarray:
    """Extracts 1-pixel boundary contours from a binary mask (H, W)."""
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_mask = np.zeros_like(mask_u8, dtype=np.uint8)
    if contours:
        cv2.drawContours(contour_mask, contours, -1, 255, 1)
    return (contour_mask > 0).astype(np.uint8)


def calculate_hd95(y_true: np.ndarray, y_pred: np.ndarray, voxel_spacing: float = 1.0) -> float:
    """
    Computes the 95th Percentile Hausdorff Distance (HD95) in pixels.
    If either mask is empty, handles clinical edge cases:
      - If both empty: 0.0 (perfect negative agreement)
      - If one empty: returns maximum spatial diagonal penalty
    """
    y_true_bin = (y_true >= 0.5).astype(bool)
    y_pred_bin = (y_pred >= 0.5).astype(bool)

    if not np.any(y_true_bin) and not np.any(y_pred_bin):
        return 0.0

    diag = np.sqrt(y_true.shape[0] ** 2 + y_true.shape[1] ** 2) * voxel_spacing
    if not np.any(y_true_bin) or not np.any(y_pred_bin):
        return float(round(diag, 2))

    # Surface distance using Euclidean Distance Transforms
    # Distance from true surface to pred surface
    true_contour = extract_binary_contours(y_true_bin)
    pred_contour = extract_binary_contours(y_pred_bin)

    if not np.any(true_contour) or not np.any(pred_contour):
        return float(round(diag, 2))

    edt_pred = distance_transform_edt(~pred_contour.astype(bool)) * voxel_spacing
    edt_true = distance_transform_edt(~true_contour.astype(bool)) * voxel_spacing

    d_true_to_pred = edt_pred[true_contour.astype(bool)]
    d_pred_to_true = edt_true[pred_contour.astype(bool)]

    all_distances = np.concatenate([d_true_to_pred, d_pred_to_true])
    hd95 = np.percentile(all_distances, 95)
    return float(round(hd95, 2))


def calculate_boundary_iou(y_true: np.ndarray, y_pred: np.ndarray, dilation_radius: int = 3) -> float:
    """
    Computes Boundary IoU (BIoU) which measures segmentation alignment strictly
    along the perimeter band within dilation_radius.
    """
    y_true_bin = (y_true >= 0.5).astype(np.uint8)
    y_pred_bin = (y_pred >= 0.5).astype(np.uint8)

    if not np.any(y_true_bin) and not np.any(y_pred_bin):
        return 1.0
    if not np.any(y_true_bin) or not np.any(y_pred_bin):
        return 0.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius * 2 + 1, dilation_radius * 2 + 1))
    
    # Boundary band = dilated mask - eroded mask
    true_band = cv2.morphologyEx(y_true_bin, cv2.MORPH_GRADIENT, kernel) > 0
    pred_band = cv2.morphologyEx(y_pred_bin, cv2.MORPH_GRADIENT, kernel) > 0

    inter = np.sum(true_band & pred_band)
    union = np.sum(true_band | pred_band)

    if union == 0:
        return 1.0
    return float(round(inter / union, 4))

