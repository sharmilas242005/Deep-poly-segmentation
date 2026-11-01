"""
Inference Engine and Visualizer for Deep Polyp Segmentation.
Generates binary masks, alpha-blended lesion overlays, contour boundaries, and latency metrics.
Strictly requires genuine trained model weights for all inference operations.
"""

import os
import time
import io
import base64
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import cv2

from ml.models.registry import create_model, MODEL_CATALOG


class PolypPredictor:
    """
    High-performance, cached model inference service with strict trained-checkpoint enforcement.
    """
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._loaded_models: Dict[str, torch.nn.Module] = {}

    def get_model(self, model_name: str, checkpoint_path: Optional[str] = None) -> torch.nn.Module:
        key = model_name.lower().strip()
        if key not in MODEL_CATALOG:
            raise ValueError(f"Unknown model '{model_name}'. Available architectures: {list(MODEL_CATALOG.keys())}")

        if key not in self._loaded_models:
            if checkpoint_path is None:
                default_ckpt = os.path.join("ml", "checkpoints", f"{key}_best.pt")
                if os.path.isfile(default_ckpt):
                    checkpoint_path = default_ckpt
                else:
                    raise FileNotFoundError(
                        f"Trained checkpoint not found for '{key}'. "
                        "Please provide trained model weights before running inference."
                    )
            model = create_model(
                key,
                device=self.device,
                checkpoint_path=checkpoint_path,
                eval_mode=True,
                require_checkpoint=True
            )
            if not getattr(model, "_is_trained", False):
                raise RuntimeError(
                    f"Model '{key}' was instantiated without trained weights. "
                    "Inference with random uninitialized weights is strictly prohibited."
                )
            self._loaded_models[key] = model
        return self._loaded_models[key]

    def preprocess_image(self, image: Image.Image, target_size: Tuple[int, int] = (256, 256)) -> Tuple[torch.Tensor, Tuple[int, int]]:
        orig_w, orig_h = image.size
        # Resize to target input resolution
        resized = image.resize(target_size, resample=Image.BILINEAR)
        # Robust conversion via buffer protocol
        np_img = np.array(resized, dtype=np.uint8)
        tensor = torch.as_tensor(np_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        tensor = tensor.unsqueeze(0).to(self.device)  # (1, 3, H, W)
        return tensor, (orig_w, orig_h)

    def predict(
        self,
        image_input: Union[str, bytes, Image.Image, np.ndarray],
        model_name: str = "segformer",
        threshold: float = 0.5,
        target_size: Tuple[int, int] = (256, 256),
        overlay_color: Tuple[int, int, int] = (0, 220, 180),  # Bright medical turquoise
        overlay_alpha: float = 0.45
    ) -> Dict[str, Any]:
        """
        Executes segmentation inference, produces binary mask, overlay, and clinical summary.
        Strictly requires trained model weights.
        """
        # 1. Parse input into PIL Image
        if isinstance(image_input, str):
            orig_pil = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            orig_pil = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            orig_pil = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            orig_pil = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image_input type: {type(image_input)}")

        orig_w, orig_h = orig_pil.size
        tensor, _ = self.preprocess_image(orig_pil, target_size)

        # 2. Model Forward Pass with precise latency profiling
        model = self.get_model(model_name)
        start_time = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # 3. Extract probabilities and threshold to binary mask
        prob_map_256 = probs.squeeze().cpu().numpy()  # (256, 256)
        # Resize probability map back to original image dimensions with Bilinear interpolation
        prob_map_orig = cv2.resize(prob_map_256, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        binary_mask_orig = (prob_map_orig >= threshold).astype(np.uint8) * 255

        # 4. Compute polyp coverage and bounding boxes
        total_pixels = orig_w * orig_h
        polyp_pixels = int(np.sum(binary_mask_orig > 0))
        area_pct = (polyp_pixels / total_pixels) * 100.0
        has_polyp = polyp_pixels > 25  # Minimum connected noise filter

        # 5. Generate Colored Alpha Overlay and Contour Highlighting
        orig_np = np.array(orig_pil)
        overlay_np = orig_np.copy()

        contours, _ = cv2.findContours(binary_mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []

        if has_polyp:
            # Color overlay for lesion pixels
            mask_bool = binary_mask_orig > 0
            for c in range(3):
                overlay_np[:, :, c] = np.where(
                    mask_bool,
                    (overlay_np[:, :, c] * (1.0 - overlay_alpha) + overlay_color[c] * overlay_alpha).astype(np.uint8),
                    overlay_np[:, :, c]
                )

            # Draw smooth contour boundaries
            cv2.drawContours(overlay_np, contours, -1, (255, 255, 255), 2)
            cv2.drawContours(overlay_np, contours, -1, overlay_color, 1)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w * h > 50:
                    bounding_boxes.append({
                        "x": int(x), "y": int(y), "width": int(w), "height": int(h),
                        "area": int(w * h)
                    })

        # 6. Format PIL outputs and Base64 Data URIs
        mask_pil = Image.fromarray(binary_mask_orig, mode="L")
        overlay_pil = Image.fromarray(overlay_np, mode="RGB")

        orig_b64 = self._image_to_data_uri(orig_pil)
        mask_b64 = self._image_to_data_uri(mask_pil)
        overlay_b64 = self._image_to_data_uri(overlay_pil)

        return {
            "model_name": model_name,
            "inference_time_ms": round(latency_ms, 2),
            "polyp_detected": bool(has_polyp),
            "polyp_area_percentage": round(area_pct, 2),
            "confidence_threshold": threshold,
            "image_dimensions": {"width": orig_w, "height": orig_h},
            "bounding_boxes": bounding_boxes,
            "detected_regions_count": len(bounding_boxes),
            "original_image_data_uri": orig_b64,
            "mask_data_uri": mask_b64,
            "overlay_data_uri": overlay_b64
        }

    def compare_models(
        self,
        image_input: Union[str, bytes, Image.Image, np.ndarray],
        model_names: Optional[List[str]] = None,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Executes multi-model side-by-side segmentation on the exact same frame.
        Verifies trained weights exist for all models before execution.
        """
        if model_names is None:
            model_names = ["segformer", "unet", "linknet"]

        # Validate trained checkpoints for all architectures before inference
        for name in model_names:
            self.get_model(name)

        results = {}
        for name in model_names:
            results[name] = self.predict(image_input, model_name=name, threshold=threshold)

        return {
            "models_evaluated": model_names,
            "results": results
        }

    @staticmethod
    def _image_to_data_uri(img: Image.Image, format: str = "PNG") -> str:
        buffered = io.BytesIO()
        img.save(buffered, format=format)
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        mime = f"image/{format.lower()}"
        return f"data:{mime};base64,{encoded}"
