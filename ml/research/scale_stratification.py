"""
Polyp Scale Stratification Utilities for Clinical Lesion Size Auditing.
Follows the clinical Paris Classification and ESGE / ASGE screening criteria:
- Diminutive / Small: Area < 5% of endoscopic frame (highest risk of false negative miss)
- Medium: 5% <= Area < 20%
- Large: Area >= 20% (elevated or broad mucosal lesions)
- Negative / Normal: No polyp pixels (control healthy tissue)
"""

import numpy as np
from typing import Tuple, Dict, Any, List


def get_polyp_scale_category(mask: np.ndarray) -> str:
    """
    Given a binary mask (H, W), calculates the percentage of lesion pixels
    and assigns the appropriate scale category.
    """
    mask_bin = (mask >= 0.5).astype(np.uint8)
    total_pixels = mask_bin.size
    lesion_pixels = int(np.sum(mask_bin))

    if lesion_pixels == 0:
        return "negative"

    pct = (lesion_pixels / total_pixels) * 100.0

    if pct < 5.0:
        return "small"
    elif pct < 20.0:
        return "medium"
    else:
        return "large"


def stratify_dataset_pairs(pairs: List[Tuple[str, str]]) -> Dict[str, List[Tuple[str, str]]]:
    """
    Groups (image_path, mask_path) pairs into scale cohorts.
    """
    from PIL import Image

    cohorts: Dict[str, List[Tuple[str, str]]] = {
        "small": [],
        "medium": [],
        "large": [],
        "negative": []
    }

    for img_p, mask_p in pairs:
        with Image.open(mask_p) as m_img:
            mask_arr = np.array(m_img.convert("L"))
        cat = get_polyp_scale_category(mask_arr)
        cohorts[cat].append((img_p, mask_p))

    return cohorts

