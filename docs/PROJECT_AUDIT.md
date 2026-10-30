# Project Audit: Deep Polyp Segmentation and Comparative Analysis

## 1. Executive Summary & Original Project Profile
- **Project Title:** *Analysis of Deep Polyp Segmentation Algorithms*
- **Academic Context:** 7th Semester B.E. Mini Project (20CSEL76), Department of Computer Science & Engineering, Global Academy of Technology (Visvesvaraya Technological University), 2023–2024.
- **Authors:** Madan S (1GA20CS071), Kishan Shetty (1GA20CS068).
- **Core Research Objective:** Evaluate and compare deep convolutional neural network architectures (specifically U-Net and LinkNet across various backbones including EfficientNetB7, ResNet34, InceptionResNetV2, DenseNet, VGG19) for semantic segmentation of gastrointestinal polyps from endoscopic images (Kvasir dataset) to aid early colorectal cancer detection.

---

## 2. Inventory of Existing Assets

| Asset / File | Format | Description / Observed Contents | Status & Issues |
| :--- | :--- | :--- | :--- |
| `UNET.ipynb` | Jupyter Notebook | U-Net training pipeline with TensorFlow/Keras and `segmentation-models`. Ran ResNet-34 backbone. | Hardcoded Colab Drive mount (`/content/drive/MyDrive/1ST DATASET`). No local path support. Cell output shows Dice ~0.7866, IoU ~0.7016 on 100 test images. |
| `LinkNet.ipynb` | Jupyter Notebook | LinkNet training pipeline with `segmentation-models`. Ran EfficientNetB7 backbone. Includes a minimal Gradio demo cell. | Path references `/content/drive/MyDrive/GI_CR_Polyp_Segmentation/bkai_igh_neopoly`. Cell 4 logged Dice ~0.8771, IoU ~0.8114. Gradio cell loads `uunet.h5` instead of LinkNet. |
| `README.docx` | MS Word Document | Brief text claiming modular files `data.py`, `train.py`, `inference.py`, and saved weights `uunet.h5` / `linknet.h5`. | Files referenced in `README.docx` (`data.py`, `train.py`, `inference.py`, `.h5` files) are completely missing from the archive. |
| Project Report PDF | PDF Document | 23-page academic report documenting literature review, methodology, and metric tables (Tables 5.4.1 & 5.4.2) for 16 model-backbone configurations. | Published table champion: U-Net + EfficientNetB7 (Dice: 0.88821, IoU: 0.8118) and LinkNet + EfficientNetB7 (Dice: 0.8742, IoU: 0.8007). |
| Trained Checkpoints (`.h5`, `.pt`) | None | No weights provided in the source bundle. | **Missing.** Weights were stored on students' private Google Drives. |
| Dataset (`images/`, `masks/`) | None | No image or mask files provided in the source bundle. | **Missing.** Dataset was mounted from Google Drive. |

---

## 3. Detailed Technical Audit & Deficiencies

### A. Environment, Framework, and Dependency Fragility
1. **Unmaintained Dependency (`segmentation-models` v1.0.1):**
   - Depends on legacy `keras-applications` and `tf.keras` from TensorFlow 1.x / 2.x prior to Keras 3.
   - Modern TensorFlow 2.16+ ships with Keras 3 as default, which completely breaks `segmentation-models` imports (`ModuleNotFoundError: No module named 'keras.utils.generic_utils'` and custom layer registration failures).
   - In modern Python (3.11 - 3.13), this legacy stack is obsolete and incompatible.
2. **Hardcoded Google Colab / Google Drive Paths:**
   - Both notebooks execute `from google.colab import drive; drive.mount('/content/drive')` and read from `/content/drive/MyDrive/...`.
   - Cannot run out-of-the-box in local environments, standard servers, or automated CI/CD.
3. **Missing Executable Scripts:**
   - `README.docx` describes a clean Python package (`data.py`, `train.py`, `inference.py`), but only two disorganized exploratory notebooks were committed.

### B. Data Integrity & Preprocessing Flaws
1. **Mask Interpolation Artifacts:**
   - Original code: `cv2.resize(mask, (256, 256))` uses default bilinear interpolation. For binary segmentation ground truths, this introduces intermediate float values (e.g., 0.1, 0.45) along polyp boundaries, corrupting the true binary mask. Proper resizing requires nearest-neighbor (`cv2.INTER_NEAREST`).
2. **Absence of Data Augmentation:**
   - While the report claims augmentation was performed, the notebook data pipeline (`tf_parse` & `tf_dataset`) applies only a simple resize and `/ 255.0` normalization. No rotations, flips, affine transforms, or color jitter were implemented.
3. **Dataset Naming Discrepancy:**
   - Report cites Kvasir-SEG (1,000 endoscopic images with ground-truth masks).
   - `UNET.ipynb` points to `1ST DATASET`.
   - `LinkNet.ipynb` points to `GI_CR_Polyp_Segmentation/bkai_igh_neopoly`.
   - Both datasets had 1,000 images, matching Kvasir-SEG's exact size.

### C. Metric Calculation and Theoretical Gaps
1. **Duplicated Identical Metrics:**
   - `sorensen_dice_coefficient` and `dice_coefficient` were implemented as separate functions in the notebooks with identical formulas: $\frac{2 |A \cap B|}{|A| + |B|}$.
   - Sørensen-Dice, Dice Similarity Coefficient (DSC), and binary F1 score are mathematically identical for binary masks.
2. **Jaccard vs. IoU Implementation Discrepancy:**
   - Jaccard Index and IoU are mathematically equivalent ($\frac{|A \cap B|}{|A \cup B|}$).
   - However, the notebook computed `intersection_over_union` on thresholded binary masks, while `jaccard_index` was computed on raw continuous predictions and masks:
     `union = np.sum(np.maximum(y_true, y_pred))`
     This produced divergent numbers (e.g. IoU 0.7016 vs. Jaccard 0.7721), confusing mathematical equivalence with implementation bugs.
3. **Fowlkes-Mallows Index Misapplication:**
   - `fowlkes_mallows_score` is a clustering metric defined as $\sqrt{\text{Precision} \cdot \text{Recall}}$. Because the background class dominates (most pixels are non-polyp), the score was artificially inflated to ~0.99, providing little diagnostic insight for lesion boundary delineation.
4. **Missing Negative Sample / Empty-Mask Handling:**
   - No epsilon or edge-case handling for images with zero polyp pixels (division by zero handling when both ground truth and prediction are empty).

### D. User Interface Limitations
1. **Basic Gradio Prototype in `LinkNet.ipynb`:**
   - Only 15 lines of UI code in Cell 6.
   - Hardcoded to load `uunet.h5` (even in the LinkNet notebook!).
   - Accepts an image and outputs only a single 256x256 binary mask.
   - No confidence threshold adjustment.
   - No colored boundary overlay on the original endoscopic frame.
   - No side-by-side multi-model comparison.
   - No inference timing or latency tracking.
   - No clinical decision-support disclaimer.

---

## 4. Reusable Components & Knowledge
- **Evaluation Split:** The deterministic 70% train / 20% validation / 10% test split logic (`random_state=42`) can be preserved and standardized so modern models can be tested against the exact same partitioning.
- **Image Resolution Baseline:** 256x256 resolution is a proven standard for rapid medical image experimentation on student hardware.
- **Historical Benchmark Reference:** The report's reported metrics for 16 backbone combinations serve as an essential historical reference baseline to compare against modern transformer and PyTorch implementations.

---

## 5. Modern Architecture Upgrade Strategy

### Phase 1: Modern ML Foundation (PyTorch)
- **Framework:** Migrate from obsolete `segmentation-models` (Keras 2) to **PyTorch 2.x** and `segmentation-models-pytorch` (smp) / Hugging Face `transformers` / `timm`.
- **Baseline Models:**
  - Modern PyTorch **U-Net** with standard CNN backbone (e.g., ResNet-34 / EfficientNet-B0/B7).
  - Modern PyTorch **LinkNet** with ResNet backbone.
- **Next-Gen Architecture:**
  - Implement **SegFormer** (hierarchical Transformer encoder with lightweight MLP decoder, e.g., `nvidia/mit-b0` or `segformer-b0-finetuned-ade-512-512` adapted for binary medical segmentation) or medical transformer alternative.
- **Loss Function:** Combined Dice Loss + Binary Cross-Entropy Loss (`DiceBCELoss`) for balanced gradient propagation on unbalanced polyp areas.
- **Metrics Module:**
  - Mathematically sound, independently unit-tested metric calculator:
    - Dice Coefficient / F1 Score
    - Intersection over Union (IoU / Jaccard)
    - Precision / Positive Predictive Value (PPV)
    - Recall / Sensitivity / True Positive Rate (TPR)
    - Specificity / True Negative Rate (TNR)
    - Latency (ms per image) and parameter count.
  - Clear documentation of mathematical identities ($\text{Dice} \equiv \text{F1} \equiv \text{Sørensen-Dice}$, $\text{IoU} \equiv \text{Jaccard} = \frac{\text{Dice}}{2 - \text{Dice}}$).

### Phase 2: Production-Grade FastAPI Backend
- RESTful endpoints:
  - `GET /api/health`: System health, GPU/CPU availability, loaded model registry.
  - `GET /api/models`: Listing of available architectures, parameters, checkpoint status.
  - `POST /api/predict`: Multi-model inference, binary mask generation, colored alpha-blended overlay generation, latency profiling.
  - `GET /api/experiments`: Historical benchmark comparison table and newly evaluated experiments.
  - `POST /api/evaluate`: Comprehensive metric evaluation on paired test images.
  - Download endpoints for predicted masks, composite overlays, and JSON/CSV reports.

### Phase 3: Research-Grade Next.js Web Interface
- Modern Next.js (App Router) + TypeScript + Tailwind CSS design system with medical imaging aesthetics.
- Multi-view layout:
  1. **Overview Dashboard:** Executive research summary, dataset health, model readiness, historical vs. modern performance overview.
  2. **Interactive Segmentation Studio:** Drag-and-drop endoscopy image upload, threshold slider, multi-model selection, side-by-side viewer (Original, Mask, Alpha Overlay, Side-by-Side Slider), latency metrics.
  3. **Multi-Model Comparison:** Synchronized visual inspection of predictions from U-Net, LinkNet, and SegFormer on the same endoscopic frame.
  4. **Evaluation & Benchmarks:** Interactive metric comparison charts (Dice, IoU, Precision, Recall, Specificity, Latency), confusion matrix pixel statistics, comparison with 2023-2024 report figures.
  5. **Training & Experiments:** Configurable training studio (batch size, learning rate, epochs, backbone), loss curve visualizer, checkpoint management.
  6. **Academic Documentation & Disclaimer:** Full methodology, dataset documentation, citations, and prominent medical AI disclaimer.
