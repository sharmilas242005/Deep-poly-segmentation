"""
Kvasir-SEG Dataset Pipeline and Integrity Verification Module.

Enforces strict medical imaging best practices:
1. Deterministic image-mask pairing and discrepancy reporting (orphaned files, corrupted formats).
2. Nearest-neighbor interpolation for ground-truth masks to eliminate interpolation boundary artifacts.
3. Explicit mask binarization (tau = 0.5 / 127).
4. Coordinated data augmentation for training (random flips, rotations) with unaugmented test/validation sets.
5. Deterministic train/validation/test split generation (70/20/10 with seed=42).
"""

import os
import glob
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class DatasetValidationReport:
    def __init__(self):
        self.total_images_found: int = 0
        self.total_masks_found: int = 0
        self.valid_pairs: List[Tuple[str, str]] = []
        self.unpaired_images: List[str] = []
        self.unpaired_masks: List[str] = []
        self.corrupted_files: List[str] = []
        self.dimension_mismatches: List[str] = []
        self.is_valid: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "total_images": self.total_images_found,
            "total_masks": self.total_masks_found,
            "paired_count": len(self.valid_pairs),
            "unpaired_images_count": len(self.unpaired_images),
            "unpaired_masks_count": len(self.unpaired_masks),
            "corrupted_files_count": len(self.corrupted_files),
            "dimension_mismatches_count": len(self.dimension_mismatches),
            "sample_pairs": self.valid_pairs[:5] if self.valid_pairs else []
        }


def validate_kvasir_dataset(dataset_dir: str) -> DatasetValidationReport:
    """
    Scans and verifies dataset integrity for Kvasir-SEG structure:
    root/
      images/ (*.jpg, *.png)
      masks/  (*.jpg, *.png)
    Also supports flat paired directory structures (e.g., sample_images/ with *_mask.png).
    """
    report = DatasetValidationReport()
    if not os.path.isdir(dataset_dir):
        return report

    images_dir = os.path.join(dataset_dir, "images")
    masks_dir = os.path.join(dataset_dir, "masks")

    image_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")

    if os.path.isdir(images_dir) and os.path.isdir(masks_dir):
        image_paths = []
        mask_paths = []
        for ext in image_exts:
            image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
            mask_paths.extend(glob.glob(os.path.join(masks_dir, ext)))

        report.total_images_found = len(image_paths)
        report.total_masks_found = len(mask_paths)

        # Index masks by filename stem (without extension)
        masks_map = {os.path.splitext(os.path.basename(p))[0]: p for p in mask_paths}
        images_map = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths}

        for stem, img_path in images_map.items():
            if stem in masks_map:
                mask_path = masks_map[stem]
                try:
                    with Image.open(img_path) as img:
                        img_size = img.size
                    with Image.open(mask_path) as msk:
                        msk_size = msk.size

                    if img_size != msk_size:
                        report.dimension_mismatches.append(f"{stem}: image {img_size} != mask {msk_size}")
                    else:
                        report.valid_pairs.append((img_path, mask_path))
                except Exception as e:
                    report.corrupted_files.append(f"{stem}: {str(e)}")
            else:
                report.unpaired_images.append(img_path)

        for stem, msk_path in masks_map.items():
            if stem not in images_map:
                report.unpaired_masks.append(msk_path)
    else:
        # Flat paired directory format (e.g. data/sample_images/ with sample_xxx.jpg and sample_xxx_mask.png)
        all_files = []
        for ext in image_exts:
            all_files.extend(glob.glob(os.path.join(dataset_dir, ext)))

        mask_files = [p for p in all_files if "_mask" in os.path.splitext(os.path.basename(p))[0].lower()]
        img_files = [p for p in all_files if p not in mask_files]

        report.total_images_found = len(img_files)
        report.total_masks_found = len(mask_files)

        masks_map = {os.path.splitext(os.path.basename(p))[0].lower().replace("_mask", ""): p for p in mask_files}
        images_map = {os.path.splitext(os.path.basename(p))[0].lower(): p for p in img_files}

        for stem, img_path in images_map.items():
            if stem in masks_map:
                mask_path = masks_map[stem]
                try:
                    with Image.open(img_path) as img:
                        img_size = img.size
                    with Image.open(mask_path) as msk:
                        msk_size = msk.size

                    if img_size != msk_size:
                        report.dimension_mismatches.append(f"{stem}: image {img_size} != mask {msk_size}")
                    else:
                        report.valid_pairs.append((img_path, mask_path))
                except Exception as e:
                    report.corrupted_files.append(f"{stem}: {str(e)}")
            else:
                report.unpaired_images.append(img_path)

        for stem, msk_path in masks_map.items():
            if stem not in images_map:
                report.unpaired_masks.append(msk_path)

    report.is_valid = len(report.valid_pairs) > 0 and len(report.dimension_mismatches) == 0
    return report


def resolve_dataset_path(preferred_path: Optional[str] = None) -> Tuple[str, DatasetValidationReport]:
    """
    Resolves the active dataset path using the strict priority:
    1. Explicit preferred_path (if provided, exists, and is valid)
    2. DATASET_ROOT_DIR environment variable (if set, exists, and is valid)
    3. ml/datasets/Kvasir-SEG (primary full Kvasir-SEG dataset)
    4. data/kvasir_seg (historical project path)
    5. data/sample_images (fallback demo cases)

    Returns:
        Tuple of (resolved_path: str, report: DatasetValidationReport)
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    candidate_paths: List[str] = []

    # Priority 0: explicit preference if given
    if preferred_path:
        cand = preferred_path if os.path.isabs(preferred_path) else os.path.join(project_root, preferred_path)
        candidate_paths.append(cand)

    # Priority 1: DATASET_ROOT_DIR environment variable
    env_dir = os.environ.get("DATASET_ROOT_DIR")
    if env_dir:
        cand = env_dir if os.path.isabs(env_dir) else os.path.join(project_root, env_dir)
        candidate_paths.append(cand)

    # Priority 2: ml/datasets/Kvasir-SEG
    candidate_paths.append(os.path.join(project_root, "ml", "datasets", "Kvasir-SEG"))

    # Priority 3: data/kvasir_seg
    candidate_paths.append(os.path.join(project_root, "data", "kvasir_seg"))

    # Priority 4: data/sample_images
    candidate_paths.append(os.path.join(project_root, "data", "sample_images"))

    fallback_result: Optional[Tuple[str, DatasetValidationReport]] = None
    for cand in candidate_paths:
        if os.path.isdir(cand):
            report = validate_kvasir_dataset(cand)
            if report.is_valid:
                return cand, report
            elif fallback_result is None:
                fallback_result = (cand, report)

    if fallback_result is not None:
        return fallback_result

    default_path = os.path.join(project_root, "ml", "datasets", "Kvasir-SEG")
    return default_path, validate_kvasir_dataset(default_path)


def get_deterministic_splits(
    pairs: List[Tuple[str, str]],
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Generates fixed, reproducible splits with seed=42 ensuring identical test sets.
    """
    total = len(pairs)
    if total == 0:
        return [], [], []

    if total <= 3:
        if total == 1:
            return pairs, pairs, pairs
        elif total == 2:
            return [pairs[0]], [pairs[1]], [pairs[1]]
        elif total == 3:
            return [pairs[0], pairs[1]], [pairs[2]], [pairs[2]]

    rng = np.random.RandomState(seed)
    indices = np.arange(total)
    rng.shuffle(indices)

    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_pairs = [pairs[i] for i in train_indices]
    val_pairs = [pairs[i] for i in val_indices]
    test_pairs = [pairs[i] for i in test_indices]

    return train_pairs, val_pairs, test_pairs


class PolypSegmentationDataset(Dataset):
    """
    PyTorch Dataset for Gastrointestinal Polyp Segmentation.
    """
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        image_size: Tuple[int, int] = (256, 256),
        is_training: bool = False,
        normalize_imagenet: bool = True
    ):
        self.pairs = pairs
        self.image_size = image_size
        self.is_training = is_training
        self.normalize_imagenet = normalize_imagenet

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]

        # 1. Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # 2. Synchronized resizing
        # Images: Bilinear interpolation
        image = image.resize(self.image_size, resample=Image.BILINEAR)
        # Masks: Strictly NEAREST NEIGHBOR to preserve crisp binary boundaries
        mask = mask.resize(self.image_size, resample=Image.NEAREST)

        # 3. Synchronized augmentations during training only
        if self.is_training:
            # Horizontal flip (50% probability)
            if torch.rand(1).item() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            # Vertical flip (30% probability)
            if torch.rand(1).item() > 0.7:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            # Random 90 deg rotation (30% probability)
            if torch.rand(1).item() > 0.7:
                angle = float(np.random.choice([90, 180, 270]))
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # 4. Conversion to tensors via buffer protocol
        np_img = np.array(image, dtype=np.uint8)
        img_tensor = torch.as_tensor(np_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask_np = (np.array(mask) >= 127).astype(np.float32)
        mask_tensor = torch.as_tensor(mask_np, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

        # 5. Normalization
        if self.normalize_imagenet:
            img_tensor = TF.normalize(
                img_tensor,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "image_path": img_path,
            "mask_path": mask_path,
            "stem": os.path.splitext(os.path.basename(img_path))[0]
        }
