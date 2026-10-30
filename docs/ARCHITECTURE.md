# Architecture Specification: Deep Polyp Segmentation & Comparative Analysis

## 1. System Overview
The Deep Polyp Segmentation and Comparative Analysis System is a modernized, full-stack, research-grade medical AI application. It provides end-to-end gastrointestinal lesion segmentation, comparative model evaluation, and interactive visual exploration for academic research and educational demonstration.

```mermaid
graph TD
    User([Endoscopist / Researcher]) -->|HTTP / REST| Frontend[Next.js + TypeScript + Tailwind CSS]
    Frontend -->|Multipart Upload / JSON| APIGateway[FastAPI Application Gateway]
    
    subgraph Backend Services
        APIGateway --> HealthRouter["/api/health"]
        APIGateway --> ModelsRouter["/api/models"]
        APIGateway --> PredictRouter["/api/predict"]
        APIGateway --> ExperimentsRouter["/api/experiments"]
        APIGateway --> EvaluateRouter["/api/evaluate"]
        
        PredictRouter --> InferenceEngine[Inference Engine (PyTorch)]
        EvaluateRouter --> EvalService[Evaluation Service]
        ExperimentsRouter --> ExpStore[Experiment & Benchmark Registry]
    end
    
    subgraph Machine Learning Core
        InferenceEngine --> ModelRegistry[Model Registry]
        ModelRegistry --> UNet[PyTorch U-Net (ResNet/CNN)]
        ModelRegistry --> LinkNet[PyTorch LinkNet (ResNet)]
        ModelRegistry --> SegFormer[SegFormer (Transformer MiT)]
        
        EvalService --> MetricsCore[Strict Mathematical Metrics Core]
        Trainer[PyTorch Trainer] --> DiceBCELoss[Dice + BCE Loss]
        DatasetLoader[Kvasir-SEG Dataset Engine] --> Preprocessing[Nearest-Neighbor Masks & Normalization]
    end
```

---

## 2. Technology Stack & Modernization Justification

| Layer | Original Legacy Stack | Modernized Stack | Technical Justification |
| :--- | :--- | :--- | :--- |
| **Deep Learning Framework** | TensorFlow 2.x + `segmentation-models` v1.0.1 | **PyTorch 2.x** (`torch`, `torchvision`, `timm`) | `segmentation-models` is unmaintained and incompatible with Keras 3 / Python 3.12+. PyTorch provides standard, stable, modular medical imaging architectures and native GPU/CPU acceleration. |
| **Model Architectures** | Legacy Keras U-Net & LinkNet | **PyTorch U-Net, LinkNet & SegFormer (Transformer)** | Retains original U-Net and LinkNet baselines while introducing modern hierarchical Transformer (SegFormer) with multi-scale self-attention. |
| **Loss Function** | Binary Cross-Entropy (`loss="binary_crossentropy"`) | **Hybrid Dice + BCE Loss (`DiceBCELoss`)** | Binary cross-entropy suffers in endoscopy where polyps occupy small pixel regions (~5-15% of the frame). Combined Dice loss directly optimizes the overlap metric. |
| **Metric Engine** | Fragmented scripts, duplicated Dice/Sørensen, unbinarized Jaccard | **Unified Mathematical Metrics Core** | Correct, unit-tested implementations of Dice, IoU, Precision, Recall, Specificity, and F1 with empty-mask ($\epsilon$) safety. |
| **Backend & API** | None (ad-hoc Colab cells) | **FastAPI + Pydantic v2 + Uvicorn** | High performance, automatic OpenAPI documentation, strict request validation, asynchronous non-blocking inference, and clean error handling. |
| **Frontend UI** | 15-line Gradio cell in LinkNet notebook | **Next.js (App Router) + TypeScript + Tailwind CSS** | State-of-the-art interactive medical research dashboard, side-by-side overlays, comparison matrix, real-time charts, and downloadable reports. |
| **Testing** | None | **PyTest Suite** | Unit tests for metric formulas, tensor dimensionalities, model forward passes, and API endpoints. |

---

## 3. Machine Learning Subsystem

### 3.1 Model Architectures
1. **U-Net (Baseline):**
   - Symmetric encoder-decoder architecture with skip connections.
   - Preserves high-resolution spatial details via concatenated encoder features.
   - Configurable CNN encoder backbone (ResNet-34 / MobileNetV2 / ConvNet).
2. **LinkNet (Efficiency Baseline):**
   - Lightweight semantic segmentation architecture designed for high inference speed.
   - Bypasses feature concatenation in favor of additive feature fusion in decoder blocks, drastically reducing parameter count and memory bandwidth.
3. **SegFormer (Modern Transformer Benchmark):**
   - Hierarchical Mix Transformer (MiT) encoder generating multi-scale features ($1/4, 1/8, 1/16, 1/32$).
   - Overlapped patch merging preserves local boundary continuity crucial for polyp edges.
   - Lightweight All-MLP decoder aggregating multi-level contextual features without complex positional encodings.

### 3.2 Mathematical Formulation of Metrics
Let $P \in \{0, 1\}^{H \times W}$ be the predicted binary mask and $Y \in \{0, 1\}^{H \times W}$ be the ground truth mask.
- **Intersection:** $TP = \sum_{i,j} P_{ij} Y_{ij}$
- **False Positives:** $FP = \sum_{i,j} P_{ij} (1 - Y_{ij})$
- **False Negatives:** $FN = \sum_{i,j} (1 - P_{ij}) Y_{ij}$
- **True Negatives:** $TN = \sum_{i,j} (1 - P_{ij}) (1 - Y_{ij})$

1. **Dice Similarity Coefficient (DSC) / F1-Score:**
   $$\text{Dice} = \frac{2 \cdot TP + \epsilon}{2 \cdot TP + FP + FN + \epsilon}$$
   *Mathematical Identity:* In binary segmentation, Sørensen-Dice, Dice, and the harmonic mean of Precision and Recall (F1-score) are strictly identical:
   $$\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \equiv \text{Dice}$$
2. **Intersection over Union (IoU) / Jaccard Index:**
   $$\text{IoU} = \frac{TP + \epsilon}{TP + FP + FN + \epsilon} \equiv \frac{\text{Dice}}{2 - \text{Dice}}$$
3. **Sensitivity / Recall:**
   $$\text{Recall} = \frac{TP + \epsilon}{TP + FN + \epsilon}$$
4. **Precision / Positive Predictive Value:**
   $$\text{Precision} = \frac{TP + \epsilon}{TP + FP + \epsilon}$$
5. **Specificity / True Negative Rate:**
   $$\text{Specificity} = \frac{TN + \epsilon}{TN + FP + \epsilon}$$

### 3.3 Data Pipeline & Integrity Guarantees
- **Mask Binarization:** Ground-truth masks are thresholded ($> 127$ or $> 0.5$) to ensure pure boolean tensors.
- **Interpolation Rules:** Images resized using Bilinear interpolation; masks resized strictly using **Nearest Neighbor** (`cv2.INTER_NEAREST` or `InterpolationMode.NEAREST`) to prevent synthetic boundary boundary artifacts.
- **Data Partitioning:** Fixed seed (`seed=42`) splitting into 70% Train, 20% Validation, 10% Test to prevent data leakage and ensure fair model-to-model benchmark comparison.

---

## 4. Backend API Design

- `GET /api/health`: System health, PyTorch version, active compute device (CPU/CUDA), and model status.
- `GET /api/models`: Catalog of available architectures, parameter counts, and checkpoint metadata.
- `POST /api/predict`: Multipart form accepting image file, selected model name, and confidence threshold. Returns base64 or URL binary mask, colored overlay, bounding box, polyp area percentage, and inference latency.
- `GET /api/experiments`: Historical benchmark comparison table (incorporating the 2023-2024 report's EfficientNetB7/ResNet34 numbers) along with newly evaluated runs.
- `POST /api/evaluate`: Evaluates model predictions against provided ground-truth masks.
- `GET /api/export/report`: Downloads formatted JSON/CSV evaluation metrics.

---

## 5. Clinical Safety & Research Disclaimer
The system enforces a mandatory visual and programmatic disclaimer:
> **RESEARCH PROTOTYPE ONLY:** This software is an academic computer-aided detection (CADe) research prototype for algorithmic comparative analysis. It is **NOT** a clinically certified medical device (FDA/CE marked) and must **NOT** be used for diagnostic, prognostic, or treatment decisions without qualified gastroenterologist oversight.
