"""
Pydantic Schemas for Deep Polyp Segmentation API.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int
    area: int


class ImageDimensions(BaseModel):
    width: int
    height: int


class PredictionResponse(BaseModel):
    model_name: str
    inference_time_ms: float
    polyp_detected: bool
    polyp_area_percentage: float
    confidence_threshold: float
    image_dimensions: ImageDimensions
    bounding_boxes: List[BoundingBox]
    detected_regions_count: int
    original_image_data_uri: str
    mask_data_uri: str
    overlay_data_uri: str
    metrics: Optional[Dict[str, float]] = None
    disclaimer: str = Field(
        default="RESEARCH PROTOTYPE ONLY: This software is for academic benchmarking and algorithmic comparison. It is NOT certified for diagnostic use without clinical specialist verification."
    )


class ModelComparisonResponse(BaseModel):
    models_evaluated: List[str]
    results: Dict[str, PredictionResponse]
    disclaimer: str = Field(
        default="RESEARCH PROTOTYPE ONLY: Results require medical-professional interpretation."
    )


class ModelInfo(BaseModel):
    id: str
    name: str
    family: str
    description: str
    input_resolution: str
    parameters: int
    parameters_formatted: str
    size_mb: float
    is_modern: bool
    historical_reference: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: float
    device: str
    pytorch_version: str
    registered_models: List[str]
    dataset_present: bool
    valid_image_mask_pairs: int


class DatasetStatusResponse(BaseModel):
    is_valid: bool
    dataset_directory: str
    paired_count: int
    unpaired_images_count: int
    unpaired_masks_count: int
    corrupted_files_count: int
    dimension_mismatches_count: int
    instructions: str


class ExperimentRecord(BaseModel):
    model_name: str
    encoder_backbone: str
    architecture_type: str
    mean_dice: float
    mean_iou: float
    precision: Optional[float] = None
    recall: Optional[float] = None
    specificity: Optional[float] = None
    mean_fm: Optional[float] = None
    latency_ms: Optional[float] = None
    parameters_m: Optional[float] = None
    source: str  # "Historical Published Reference (2023-2024)" vs "Newly Evaluated PyTorch"
    status: str


class TrainStartRequest(BaseModel):
    model_name: str = "segformer"
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-4
    dataset_source: str = "auto"
    early_stopping_patience: int = 5


class TrainingHistoryStep(BaseModel):
    epoch: int
    train_loss: float
    val_loss: float
    val_dice: float
    val_iou: float
    learning_rate: float


class TrainingStatusResponse(BaseModel):
    status: str  # "idle", "running", "completed", "stopped", "failed"
    model_name: Optional[str] = None
    current_epoch: int = 0
    total_epochs: int = 0
    progress_percent: float = 0.0
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    val_dice: Optional[float] = None
    val_iou: Optional[float] = None
    learning_rate: Optional[float] = None
    best_dice: Optional[float] = None
    elapsed_time_sec: float = 0.0
    history: List[TrainingHistoryStep] = []
    error_message: Optional[str] = None
    checkpoint_path: Optional[str] = None
    dataset_used: Optional[str] = None
    dataset_pairs: int = 0


class EvaluationResponse(BaseModel):
    model_name: str
    inference_time_ms: float
    has_ground_truth: bool
    metrics: Optional[Dict[str, float]] = None
    polyp_detected: bool
    polyp_area_percentage: float
    detected_regions_count: int
    original_image_data_uri: str
    mask_data_uri: str
    ground_truth_mask_data_uri: Optional[str] = None
    overlay_data_uri: str
    disclaimer: str = Field(
        default="RESEARCH PROTOTYPE ONLY: This software is for academic benchmarking and algorithmic comparison. It is NOT certified for diagnostic use without clinical specialist verification."
    )

