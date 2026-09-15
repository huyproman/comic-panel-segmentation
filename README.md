# Comic Panel Extraction

Official implementation of the paper:  
**"Comic Panel Extraction using U-Net++ Deep Learning Model and Line Segment Analysis"**

---

## Overview

Automatic comic panel extraction adapts page-based comics to digital displays. Comics feature diverse artistic styles, non-rectangular panels, black backgrounds, and speech balloons or characters that overlap panel boundaries.

This repository provides a standalone implementation of the proposed hybrid panel extraction framework:
1. **Neural Segmentation**: A **U-Net++** architecture trained from scratch produces initial panel probability maps without rigid geometric assumptions.
2. **Line Segment Analysis (LSA)**: Extracts structural boundary lines using Probabilistic Hough Transform, merges parallel segments, clusters intersections into corners using Mean-Shift, and validates candidate connections via Bresenham sampling.
3. **Geometric Filtering (GF)**: Cleans residual artifacts using component size constraints to output final panel instances.
4. **Instance Evaluation**: Evaluates extracted panels against ground truth using bipartite matching (Hungarian algorithm) and computes panel and page-level metrics.

---

## Installation

### Prerequisites
- Python 3.10 or higher
- CUDA-compatible GPU recommended for training

### Install from Source
```bash
git clone https://github.com/huyproman/comic-panel-segmentation.git
cd comic-panel-segmentation
pip install -e .
```

---

## Quickstart

### 1. Inference
Run panel extraction on an image. By default, both neural segmentation and line-segment post-processing are executed:
```bash
comics-panel-extraction infer page.jpg --checkpoint path/to/checkpoint.keras --output final_mask.png
```

To also inspect the raw neural segmentation before post-processing:
```bash
comics-panel-extraction infer page.jpg --checkpoint path/to/checkpoint.keras --output final_mask.png --raw-output raw_mask.png
```

To run only the neural model without geometric post-processing:
```bash
comics-panel-extraction infer page.jpg --checkpoint path/to/checkpoint.keras --output raw_mask.png --raw-only
```

### 2. Standalone Post-Processing
Apply line-segment post-processing directly to an existing binary segmentation mask:
```bash
comics-panel-extraction postprocess raw_mask.png --output refined_mask.png
```

### 3. Evaluation
Evaluate predicted binary masks against ground truth masks:
```bash
comics-panel-extraction evaluate --predictions preds/ --ground-truth gt/ --output-json metrics.json
```

### 4. Training
Train the U-Net++ segmentation model from scratch using a dataset directory containing `images/` and `masks/`:
```bash
comics-panel-extraction train --dataset path/to/dataset --config configs/default.yaml --epochs 150 --batch-size 8 --output-dir runs/train
```

---

## Method Specifications

### Model Architecture
- **Backbone**: U-Net++ without deep supervision (single output tensor).
- **Encoder Channels**: 32, 64, 128, 256, 512.
- **Layers**: Double Conv2D + Batch Normalization + ReLU per block; dense nested skip pathways.
- **Output**: 1x1 Conv2D with Sigmoid activation for binary panel classification.
- **Input / Output Resolution**: 448x448x3 (Input) -> 448x448x1 (Output probability).
- **Total Parameters**: 9,056,769.

### Training Pipeline
- **Loss**: Binary Cross-Entropy (BCE) + Soft Dice Loss.
- **Optimizer**: AdamW with Warmup Cosine Decay schedule.

### Line Segment Analysis
- **Edge Detection**: Morphological opening (20x20) followed by Canny edge detection (50/100) and bridge morphology.
- **Line Detection**: Probabilistic Hough Transform (rho=2, theta=pi/180, threshold=25, minLineLength=10, maxLineGap=30).
- **Grouping & Clustering**: Distance features relative to reference geometry (384, 384); intersections clustered via Mean-Shift (inner bandwidth=13, border bandwidth=10, clipping to [0, 400]).
- **Bresenham Validation**: Edge support ratio >= 0.75, minimum Euclidean length > 40 px; carved with thickness 3.
- **Filtering**: Retains connected components strictly greater than 1000 px.

---

## Reproducibility Note

This repository reconstructs the model architecture, training pipeline, line-segment analysis, and evaluation framework as described in the paper. The exact trained model instances reported in the manuscript are not bundled due to storage and distribution constraints. Models can be trained from scratch using the provided training pipeline and configuration.

---

## Citation

```bibtex
@inproceedings{nguyen2025comic,
  title={Comic Panel Extraction using U-Net++ Deep Learning Model and Line Segment Analysis},
  author={Nguyen, Quang Huy and Le, Linh Long and Nghiem, Thi Phuong and Pham, Thai Son and Nguyen, Tuan Khai and Nguyen, Hai Dang and Tran, Giang Son},
  booktitle={Proceedings of the International Conference on Research in Intelligent and Computing in Engineering (RIVF)},
  year={2025}
}
```
