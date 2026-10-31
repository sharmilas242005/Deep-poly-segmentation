# Deep Polyp Segmentation: Comparative Clinical Research Platform

[![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js 16](https://img.shields.io/badge/Next.js-16%20(Turbopack)-000000?logo=next.js&logoColor=white)](https://nextjs.org/)
[![Tailwind CSS 4](https://img.shields.io/badge/Tailwind-CSS%204-38B2AC?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Tests Passing](https://img.shields.io/badge/Tests-18%2F18%20Passing-success)](backend/tests/)

> **Academic Modernization & Capstone Upgrade**  
> Based on the VTU (Visvesvaraya Technological University) 2023–2024 B.E. Computer Science & Engineering project:  
> *"Analysis of Deep Polyp Segmentation Algorithms"*

---

## 1. Project Overview & Upgrade Rationale

Colorectal polyps are mucosal precursors to colorectal cancer (CRC). Early endoscopic identification and precise boundary segmentation during colonoscopy significantly reduce patient mortality.

This repository modernizes the original academic codebase from legacy, fragile notebook implementations into an end-to-end, production-ready, demonstrable research platform.

### Modernization Matrix

| Dimension | Original Academic Project (2023–2024) | Modern Upgraded System (2025–2026) |
|---|---|---|
| **Deep Learning Framework** | TensorFlow 2.x / Keras (Legacy) | **PyTorch 2.3+** (Modular, CPU & CUDA ready) |
| **Segmentation Library** | Unmaintained `segmentation-models` v1.0.1 | **Pure PyTorch Architectures** with clean module inheritance |
| **Architectures** | Monolithic U-Net & LinkNet notebooks | **U-Net**, **LinkNet**, and **SegFormer** (Vision Transformer) |
| **Loss Formulation** | Standard BCE / Jaccard loss | Hybrid **`DiceBCELoss`** with numerical stability clamp |
| **Dataset Pipeline** | Hardcoded Google Drive paths (`/content/drive/...`) | Deterministic **Kvasir-SEG Validator** (70/20/10 split, seed=42) |
| **Inference Engine** | Slow, repeated model loading in Gradio | **Cached Predictor Engine** with contour extraction & overlays |
| **Backend API** | Inline Gradio script in notebook | High-throughput **FastAPI + Pydantic v2** REST service |
| **Frontend UI** | Basic Gradio web blocks | Interactive **Next.js 16 + Tailwind CSS 4 + Recharts** Studio |
| **Automated Testing** | 0 unit or integration tests | **18 Automated Tests** (Metrics, Models, API Endpoints) |
| **Safety Governance** | No medical safety disclosure | Prominent **Academic CADe Research Disclaimer** |

---

## 2. Supported Architectures

```
                    ┌────────────────────────────────────────────────────────┐
                    │               Endoscopy Frame (RGB)                    │
                    └──────────────────────────┬─────────────────────────────┘
                                               │
               ┌───────────────────────────────┼───────────────────────────────┐
               ▼                               ▼                               ▼
    ┌──────────────────────┐        ┌──────────────────────┐        ┌──────────────────────┐
    │     Classic U-Net    │        │       LinkNet        │        │      SegFormer       │
    ├──────────────────────┤        ├──────────────────────┤        ├──────────────────────┤
    │ • Biomedical encoder │        │ • Residual blocks    │        │ • MiT Transformer    │
    │ • Concatenation skips│        │ • Additive skips     │        │ • Overlapped Patches │
    │ • Precise boundaries │        │ • Lightweight params │        │ • Multi-scale All-MLP│
    └──────────┬───────────┘        └──────────┬───────────┘        └──────────┬───────────┘
               │                               │                               │
               └───────────────────────────────┼───────────────────────────────┘
                                               ▼
                              ┌───────────────────────────────────┐
                              │ Predicted Mask & Heatmap Overlay  │
                              └───────────────────────────────────┘
```

1. **Biomedical U-Net** (`ml/models/unet.py`):
   - Symmetrical contracting (encoder) and expanding (decoder) paths.
   - Feature channels are concatenated via skip connections to preserve fine spatial endoscopic textures.
2. **LinkNet** (`ml/models/linknet.py`):
   - Uses residual encoder blocks and additive skip connections.
   - Decoders bypass heavy channel concatenations, significantly reducing memory bandwidth and parameters.
3. **SegFormer** (`ml/models/segformer.py`):
   - Multi-scale Mix Transformer (MiT) encoder using overlapped patch embeddings without explicit positional codes.
   - Lightweight All-MLP decoder that fuses multi-level feature maps ($1/4, 1/8, 1/16, 1/32$) to capture both global mucosal topology and small sessile polyp borders.

---

## 3. Mathematical Metric Formulation

All metrics are implemented with strict mathematical precision in [`ml/evaluation/metrics.py`](ml/evaluation/metrics.py):

$$\text{Dice Coefficient (F1)} = \frac{2 \cdot |Y \cap \hat{Y}|}{|Y| + |\hat{Y}|} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

$$\text{Jaccard Index (IoU)} = \frac{|Y \cap \hat{Y}|}{|Y \cup \hat{Y}|} = \frac{TP}{TP + FP + FN} = \frac{\text{Dice}}{2 - \text{Dice}}$$

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall (Sensitivity)} = \frac{TP}{TP + FN}, \quad \text{Specificity} = \frac{TN}{TN + FP}$$

---

## 4. Benchmark Performance Comparison

### 4.1 Historical Published Results (VTU Report 2023–2024 Reference)
*Evaluated across 16 encoder-decoder backbone variations on the Kvasir-SEG test split:*

| Model Architecture | Encoder Backbone | Parameters (M) | Test Mean Dice | Test Mean IoU |
|---|---|---|---|---|
| **U-Net** | EfficientNet-B7 | 64.0M | **0.8882** | **0.8118** |
| **U-Net** | EfficientNet-B4 | 19.3M | 0.8824 | 0.8041 |
| **U-Net** | ResNet-34 | 21.8M | 0.7866 | 0.7016 |
| **U-Net** | VGG-19 | 20.0M | 0.7712 | 0.6845 |
| **LinkNet** | EfficientNet-B7 | 64.0M | 0.8654 | 0.7812 |
| **LinkNet** | ResNet-34 | 21.3M | 0.7745 | 0.6912 |
| **LinkNet** | VGG-19 | 19.8M | 0.7589 | 0.6724 |

### 4.2 Evaluated PyTorch Modern Runtimes
*Real-time CPU inference benchmarked on standardized $256 \times 256$ input frames:*

| Model Architecture | Parameters | CPU Latency | Polyp Detection | Boundary Crispness |
|---|---|---|---|---|
| **SegFormer (MiT-B0)** | **3.71M** | **~65 ms** | Excellent | Multi-scale boundary preservation |
| **LinkNet (ResNet-style)** | 11.53M | ~154 ms | Very Good | Fast additive decoder |
| **Classic U-Net** | 13.40M | ~200 ms | Good | High-contrast mucosal edges |

---

## 5. Quick Start: How to Run

### Method 1: One-Click Windows Launcher (Recommended)
Double-click [`run_project.bat`](run_project.bat) or run in PowerShell:
```powershell
.\run_project.bat
```
*This automatically launches the FastAPI backend on port 8000, starts the Next.js frontend on port 3000, and opens your default browser.*

---

### Method 2: Manual Terminal Execution

#### 1. Start the FastAPI Backend
```bash
# In project root
.\.venv\Scripts\uvicorn.exe backend.app.main:app --host 127.0.0.1 --port 8000
```
- API Health Check: `http://127.0.0.1:8000/api/health`
- Interactive Swagger Docs: `http://127.0.0.1:8000/docs`

#### 2. Start the Next.js Web Studio
```bash
cd frontend
npm run start
# Or for live hot-reloading development:
# npm run dev
```
- Open in Browser: `http://localhost:3000`

---

## 6. Running Automated Tests

Run the complete 18-test suite covering mathematical metric stability, neural network tensor dimensions, and FastAPI endpoints:
```bash
.\.venv\Scripts\pytest.exe backend/tests/test_metrics.py backend/tests/test_models.py backend/tests/test_api.py -v
```

**Expected Output:**
```
backend/tests/test_metrics.py::test_perfect_prediction_metrics PASSED
backend/tests/test_metrics.py::test_zero_prediction_metrics PASSED
backend/tests/test_metrics.py::test_dice_iou_mathematical_relationship PASSED
backend/tests/test_metrics.py::test_metric_tracker_accumulation PASSED
backend/tests/test_metrics.py::test_losses_edge_cases PASSED
backend/tests/test_metrics.py::test_dice_bce_loss_gradient PASSED
backend/tests/test_models.py::test_unet_forward_pass PASSED
backend/tests/test_models.py::test_linknet_forward_pass PASSED
backend/tests/test_models.py::test_segformer_forward_pass PASSED
backend/tests/test_models.py::test_model_registry PASSED
backend/tests/test_models.py::test_model_save_and_load PASSED
backend/tests/test_api.py::test_health_endpoint PASSED
backend/tests/test_api.py::test_models_endpoint PASSED
backend/tests/test_api.py::test_experiments_endpoint PASSED
backend/tests/test_api.py::test_samples_endpoint PASSED
backend/tests/test_api.py::test_dataset_status_endpoint PASSED
backend/tests/test_api.py::test_predict_endpoint PASSED
backend/tests/test_api.py::test_compare_endpoint PASSED

======================== 18 passed in 10.34s ========================
```

---

## 7. Training a Custom Model

You can train any architecture on your own dataset or Kvasir-SEG directory:
```bash
# Train SegFormer for 20 epochs
.\.venv\Scripts\python.exe -m ml.training.trainer --model segformer --epochs 20 --batch-size 8 --lr 0.0003

# Train U-Net
.\.venv\Scripts\python.exe -m ml.training.trainer --model unet --epochs 25 --batch-size 8

# Train LinkNet
.\.venv\Scripts\python.exe -m ml.training.trainer --model linknet --epochs 25 --batch-size 8
```
*Trained checkpoints are automatically saved to `ml/checkpoints/<model_name>_best.pt`.*

---

## 8. REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | `GET` | PyTorch version, active hardware device (CPU/CUDA), model registry count. |
| `/api/models` | `GET` | Metadata, parameter counts, and theoretical description for all 3 models. |
| `/api/samples` | `GET` | Pre-bundled endoscopy test frames with ground-truth masks. |
| `/api/predict` | `POST` | Multipart image upload; returns mask Data URI, contour overlay, bounding boxes, latency. |
| `/api/compare` | `POST` | Evaluates SegFormer, U-Net, and LinkNet side-by-side on the uploaded frame. |
| `/api/experiments` | `GET` | Unified 16-model benchmark database from 2023–2024 VTU report vs modern PyTorch. |
| `/api/dataset/status` | `GET` | Integrity audit of dataset folder (valid pairs, missing masks, corrupted files). |

---

## 9. Academic Presentation & Viva Guide

When presenting this project to your professor and external examiners, highlight the following core engineering achievements:

1. **Why replace TensorFlow 2.x / segmentation-models with PyTorch?**
   - The legacy `segmentation-models` v1.0.1 package relied on frozen Keras 2 APIs that break in modern Python 3.11+ environments.
   - The upgraded PyTorch 2.3+ implementation provides native tensor operations, transparent PyTorch module trees, and zero external dependency risk.
2. **Why add SegFormer alongside U-Net and LinkNet?**
   - U-Net and LinkNet rely strictly on local convolutional receptive fields. While effective, they can struggle with variable polyp sizes, specular light reflections, and faint flat lesions.
   - SegFormer introduces Mix Vision Transformers (MiT) with multi-head self-attention, capturing global mucosal context across the colon wall while maintaining a lightweight parameter footprint (3.71M parameters).
3. **What is the clinical significance of Dice vs. IoU?**
   - Dice is equivalent to the F1 score and gives higher weight to true positives, reflecting diagnostic overlap.
   - IoU (Jaccard Index) penalizes false positives and false negatives more severely ($\text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}}$), providing a stringent measure of surgical excision boundary accuracy.
4. **Clinical Safety & Regulatory Disclosure:**
   - Always state clearly: *"This is a Computer-Assisted Detection (CADe) research prototype for academic algorithmic benchmarking. It is NOT an FDA/CE-cleared medical device and must never replace clinical histopathology or colonoscopist diagnosis."*

---

## 10. Repository File Structure

```
Deep poly segmentation/
├── backend/
│   ├── app/
│   │   ├── api/endpoints.py       # FastAPI REST endpoints
│   │   ├── core/config.py         # Path configurations & settings
│   │   ├── schemas/segmentation.py# Pydantic v2 data models
│   │   └── main.py                # App entrypoint & CORS middleware
│   ├── tests/
│   │   ├── test_api.py            # API endpoint unit tests
│   │   ├── test_metrics.py        # Mathematical metric verification
│   │   └── test_models.py         # Model tensor shape & forward pass tests
│   └── requirements.txt           # Pinned backend dependencies
├── data/
│   └── sample_images/             # Endoscopy test frames & ground-truth masks
├── docs/
│   ├── ARCHITECTURE.md            # Detailed engineering & math architecture
│   ├── EXPERIMENTS.md             # Complete 16-model VTU historical benchmarks
│   └── PROJECT_AUDIT.md           # Comprehensive legacy codebase audit
├── frontend/
│   ├── app/
│   │   ├── page.tsx               # 6-view clinical research studio
│   │   ├── layout.tsx             # Root layout with fonts & dark theme
│   │   └── globals.css            # Tailwind CSS styling tokens
│   ├── next.config.ts             # API proxy rewrite to backend
│   └── package.json               # Next.js 16 dependencies
├── ml/
│   ├── checkpoints/               # Trained .pt checkpoint weights
│   ├── datasets/
│   │   ├── kvasir_dataset.py      # Split generator & validator
│   │   └── sample_generator.py    # Test endoscopy frame generator
│   ├── evaluation/
│   │   └── metrics.py             # Mathematically verified Dice/IoU/F-measure
│   ├── inference/
│   │   └── predictor.py           # Cached inference engine & overlay generator
│   ├── models/
│   │   ├── unet.py                # PyTorch U-Net architecture
│   │   ├── linknet.py             # PyTorch LinkNet architecture
│   │   ├── segformer.py           # PyTorch SegFormer architecture
│   │   └── registry.py            # Model catalog & factory
│   └── training/
│       ├── losses.py              # Hybrid DiceBCELoss
│       └── trainer.py             # PyTorch training loop with AdamW & scheduler
├── run_project.bat                # Windows one-click batch launcher
├── run_project.ps1                # PowerShell one-click runner
├── LinkNet.ipynb                  # Original 2023-2024 student notebook (Preserved)
├── UNET.ipynb                     # Original 2023-2024 student notebook (Preserved)
├── README.docx                    # Original 2023-2024 report docx (Preserved)
└── README.md                      # Modern technical documentation
```

---

## 11. License & Academic Citation
Developed for academic benchmarking and educational research in Medical Computer Vision and Deep Learning.  
When referencing this work:
```bibtex
@misc{deep_polyp_segmentation_2026,
  title={Analysis and Modernization of Deep Polyp Segmentation Algorithms},
  author={VTU CSE Academic Capstone Team},
  year={2026},
  note={PyTorch, FastAPI, and Next.js Implementation}
}
```
# Deep-poly-segmentation