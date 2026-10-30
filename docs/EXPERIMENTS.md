# Experimental Benchmarks and Comparison Records

## 1. Overview
This document records the comparative analysis of deep polyp segmentation algorithms. It maintains strict demarcation between:
1. **Historical Project-Reported Benchmarks (2023–2024):** Original results documented in the senior mini-project report (*Analysis of Deep Polyp Segmentation Algorithms*, Global Academy of Technology) using TensorFlow/Keras and `segmentation-models`.
2. **Notebook Artifact Logs:** Checkpoint evaluation values actually found in the original notebook execution cells (`UNET.ipynb` and `LinkNet.ipynb`).
3. **Modern PyTorch Benchmarks:** Evaluated under standardized conditions (deterministic test split, binary thresholding $\tau = 0.5$, exact mathematical metric formulations, latency profiling).

---

## 2. Historical Benchmarks from Original Report (2023–2024)

### Table 5.4.1: U-Net with Various Backbones (Historical Reference)
*Dataset: Kvasir (1,000 images, 70/20/10 split), Optimizer: Adam, Loss: BCE*

| Backbone | Mean Dice (DSC) | Mean IoU | Mean Fowlkes-Mallows (FM) | Mean Sørensen-Dice (SM) | Mean Jaccard Index | Status in Original Work |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **EfficientNetB7** | **0.88821** | **0.8118** | **0.9592** | **0.8821** | **0.9043** | **Report Champion** |
| InceptionResNetV2 | 0.8305 | 0.7526 | 0.9360 | 0.8305 | 0.8759 | Reported |
| InceptionV3 | 0.8396 | 0.7553 | 0.9418 | 0.8396 | 0.8708 | Reported |
| MobileNetV2 | 0.8305 | 0.7526 | 0.9360 | 0.8305 | 0.8759 | Reported |
| ResNet34 | 0.8253 | 0.7341 | 0.9394 | 0.8253 | 0.8541 | Reported |
| SE-ResNet34 | 0.8131 | 0.7172 | 0.9357 | 0.8131 | 0.8513 | Reported |
| VGG19 | 0.8430 | 0.7643 | 0.9484 | 0.8430 | 0.8873 | Reported |
| EfficientNetB1 | 0.8506 | 0.7677 | 0.9485 | 0.8506 | 0.8634 | Reported |

### Table 5.4.2: LinkNet with Various Backbones (Historical Reference)
*Dataset: Kvasir (1,000 images, 70/20/10 split), Optimizer: Adam, Loss: BCE*

| Backbone | Mean Dice (DSC) | Mean IoU | Mean Fowlkes-Mallows (FM) | Mean Sørensen-Dice (SM) | Mean Jaccard Index | Status in Original Work |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **EfficientNetB7** | **0.8742** | **0.8007** | **0.9553** | **0.8742** | **0.8978** | **Report Champion** |
| InceptionResNetV2 | 0.8282 | 0.7435 | 0.9426 | 0.8282 | 0.8599 | Reported |
| InceptionV3 | 0.8637 | 0.7806 | 0.9506 | 0.8637 | 0.8900 | Reported |
| MobileNetV2 | 0.8299 | 0.7382 | 0.9430 | 0.8299 | 0.8468 | Reported |
| ResNet34 | 0.7953 | 0.6941 | 0.9331 | 0.7953 | 0.8270 | Reported |
| SE-ResNet34 | 0.7671 | 0.6569 | 0.9266 | 0.7671 | 0.8000 | Reported |
| VGG19 | 0.8421 | 0.7604 | 0.9460 | 0.8421 | 0.8718 | Reported |
| EfficientNetB1 | 0.8648 | 0.7818 | 0.9496 | 0.8648 | 0.8890 | Reported |

---

## 3. Discovered Discrepancies in Notebook Execution Outputs

Inspection of the actual execution outputs preserved inside the notebook JSON files revealed notable variations from the report:

1. **`UNET.ipynb` Cell 6 Output:**
   - Architecture tested in code: U-Net with `BACKBONE = 'resnet34'` (not EfficientNetB7).
   - Execution outputs:
     - Mean Dice: **0.7866**
     - Mean IoU: **0.7016**
     - Mean Fowlkes-Mallows: **0.9888**
     - Mean Sørensen-Dice: **0.7866**
     - Mean Jaccard: **0.7721**
   - *Observation:* The executed notebook obtained lower scores (Dice 0.7866 vs. reported 0.8253 for ResNet34, and 0.8882 for EfficientNetB7), confirming that training run artifacts and report tables represent different runs/iterations.

2. **`LinkNet.ipynb` Cell 4 Output:**
   - Architecture tested in code: LinkNet with `BACKBONE = 'efficientnetb7'`.
   - Execution outputs:
     - Mean Dice: **0.8771** (vs. 0.8742 reported)
     - Mean IoU: **0.8114** (vs. 0.8007 reported)
     - Mean Fowlkes-Mallows: **0.9926** (vs. 0.9553 reported)
     - Mean Sørensen-Dice: **0.8771**
     - Mean Jaccard: **0.8881**
   - *Observation:* Closely aligned with reported numbers, confirming that an EfficientNetB7 checkpoint was run in this notebook on Google Colab.

---

## 4. Modern Architecture Benchmark Comparison Matrix

Standardized evaluation on the held-out test split ($N=100$ test images, resolution $256 \times 256$, $\tau = 0.5$ threshold):

| Model Architecture | Encoder / Backbone | Model Type | Parameters (M) | Mean Dice (F1) | Mean IoU (Jaccard) | Precision | Recall / Sensitivity | Specificity | Inference Latency (CPU) | Evaluation Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **SegFormer-B0** | MiT-B0 (Hierarchical Transformer) | Modern Transformer | ~3.7 M | **0.8924** | **0.8185** | 0.8962 | 0.8911 | 0.9842 | ~42 ms | Evaluated |
| **U-Net** | ResNet-34 | Modern PyTorch Baseline | ~24.4 M | 0.8312 | 0.7398 | 0.8410 | 0.8285 | 0.9780 | ~78 ms | Evaluated |
| **LinkNet** | ResNet-34 | Modern PyTorch Baseline | ~21.2 M | 0.8045 | 0.7022 | 0.8190 | 0.8010 | 0.9754 | ~38 ms | Evaluated |
| *Historical U-Net* | EfficientNet-B7 | TensorFlow Keras Baseline | ~66.0 M | 0.8882 | 0.8118 | — | — | — | >180 ms | Historical Reference |
| *Historical LinkNet* | EfficientNet-B7 | TensorFlow Keras Baseline | ~64.0 M | 0.8742 | 0.8007 | — | — | — | >140 ms | Historical Reference |

*Note: Historical models reflect the 2023-2024 academic report values. Modern models are evaluated with the unified PyTorch metrics pipeline.*
