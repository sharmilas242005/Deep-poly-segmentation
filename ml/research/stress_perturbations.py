"""
Controlled Endoscopic Stress Perturbations for Robustness Auditing.
Simulates common clinical colonoscopy imaging artifacts:
1. Specular Highlights / Light Glare (mucosal fluid reflection)
2. Peristaltic Motion Blur (scope movement / bowel wall contractions)
3. Fluid Turbidity & Contrast Attenuation (poor bowel preparation / fluid pooling)
"""

import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Union


def inject_specular_glare(
    image: Union[np.ndarray, Image.Image],
    num_spots: int = 4,
    max_radius: int = 18,
    intensity: float = 0.95,
    seed: int = 42
) -> np.ndarray:
    """
    Simulates high-intensity specular reflection spots typical of wet mucosal tissue under colonoscope light.
    """
    rng = np.random.default_rng(seed)
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"), dtype=np.uint8)
    else:
        img_np = np.array(image, dtype=np.uint8).copy()

    h, w, c = img_np.shape
    glare_mask = np.zeros((h, w), dtype=np.float32)

    for _ in range(num_spots):
        cx = rng.integers(int(w * 0.15), int(w * 0.85))
        cy = rng.integers(int(h * 0.15), int(h * 0.85))
        r = rng.integers(6, max_radius)
        cv2.circle(glare_mask, (cx, cy), r, 1.0, -1)

    # Soften glare edges with Gaussian blur
    glare_soft = cv2.GaussianBlur(glare_mask, (15, 15), 5)
    glare_3d = np.repeat(glare_soft[:, :, np.newaxis], 3, axis=2)

    # Blend saturated white highlight onto image
    white = np.full_like(img_np, 255, dtype=np.float32)
    corrupted = img_np.astype(np.float32) * (1.0 - glare_3d * intensity) + white * (glare_3d * intensity)
    return np.clip(corrupted, 0, 255).astype(np.uint8)


def inject_motion_blur(
    image: Union[np.ndarray, Image.Image],
    kernel_size: int = 15,
    angle_degrees: float = 45.0
) -> np.ndarray:
    """
    Simulates linear directional motion blur caused by colonoscope advancement or rapid peristalsis.
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"), dtype=np.uint8)
    else:
        img_np = np.array(image, dtype=np.uint8).copy()

    # Create directional motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = (kernel_size - 1) / 2.0
    rad = np.deg2rad(angle_degrees)
    dx = np.cos(rad)
    dy = np.sin(rad)

    for i in range(kernel_size):
        offset = i - center
        x = int(round(center + offset * dx))
        y = int(round(center + offset * dy))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0

    k_sum = kernel.sum()
    if k_sum > 0:
        kernel /= k_sum
    else:
        kernel[int(center), int(center)] = 1.0

    blurred = cv2.filter2D(img_np, -1, kernel)
    return blurred.astype(np.uint8)


def inject_fluid_turbidity(
    image: Union[np.ndarray, Image.Image],
    contrast_factor: float = 0.65,
    turbidity_tint: Tuple[float, float, float] = (0.9, 0.85, 0.6)  # Slight yellowish/amber bile tint
) -> np.ndarray:
    """
    Simulates optical attenuation and chromatic scatter from turbid bowel preparation fluid.
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"), dtype=np.uint8)
    else:
        img_np = np.array(image, dtype=np.uint8).copy()

    img_float = img_np.astype(np.float32) / 255.0
    # Contrast reduction around mucosal midtone
    mean_val = 0.5
    degraded = mean_val + (img_float - mean_val) * contrast_factor
    # Chromatic tinting
    for c in range(3):
        degraded[:, :, c] = degraded[:, :, c] * turbidity_tint[c]

    degraded_u8 = np.clip(degraded * 255.0, 0, 255).astype(np.uint8)
    return degraded_u8

