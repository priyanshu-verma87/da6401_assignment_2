# DA6401 Assignment 2 - Visual Perception Pipeline

This project implements a **complete multi-task visual perception pipeline** using **PyTorch**, trained on the **Oxford-IIIT Pet Dataset**. The pipeline performs breed classification, object localization, and semantic segmentation in a unified architecture.

The project integrates **Weights & Biases (W&B)** for experiment tracking and analysis.

---

# Features

- VGG11 encoder built **from scratch** using PyTorch primitives
- **Custom Dropout** layer implemented without `nn.Dropout`
- **Custom IoU Loss** implemented without external libraries
- **U-Net style segmentation** with transposed convolutions
- **Multi-task learning** — single forward pass for all 3 tasks
- Transfer learning with frozen / partial / full fine-tuning strategies
- W&B experiment tracking for all 8 report sections

---

# Project Structure

```
da6401_assignment_2/
├── checkpoints/
│   ├── checkpoints.md
│
├── data/
│   └── pets_dataset.py         ← Oxford-IIIT Pet dataset loader
│
├── losses/
│   ├── __init__.py
│   ├── iou_loss.py             ← Custom IoU Loss
│   └── dice_loss.py            ← Dice Loss for segmentation
│
├── models/
│   ├── __init__.py
│   ├── layers.py               ← Custom Dropout
│   ├── vgg11.py                ← VGG11 Encoder backbone
│   ├── classification.py       ← Task 1: VGG11Classifier
│   ├── localization.py         ← Task 2: VGG11Localizer
│   ├── segmentation.py         ← Task 3: VGG11UNet
│   └── multitask.py            ← Task 4: MultiTaskPerceptionModel
│
│
├── inference.py                ← Run pipeline on single image
├── train.py                    ← Training entrypoint for all 4 tasks
├── README.md
└── requirements.txt
```

---

# Dataset

**Oxford-IIIT Pet Dataset** — 37 pet breed categories (~200 images/class)

Each image has three annotations:

- **Class label** — breed name (37 classes)
- **Bounding box** — head ROI in `[cx, cy, w, h]` normalised format
- **Trimap mask** — pixel-level segmentation (foreground / background / boundary)

Download:

```bash
# Images
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz

# Annotations
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
```

---

# Model Architecture

## Task 1 — VGG11Classifier

```
Input (3, 224, 224)
        ↓
VGG11 Encoder (blocks 1-5)
    Block 1: Conv(64)              → BN → ReLU → MaxPool
    Block 2: Conv(128)             → BN → ReLU → MaxPool
    Block 3: Conv(256) → Conv(256) → BN → ReLU → MaxPool
    Block 4: Conv(512) → Conv(512) → BN → ReLU → MaxPool
    Block 5: Conv(512) → Conv(512) → BN → ReLU → MaxPool
        ↓
Classification Head
    AdaptiveAvgPool → Flatten
    FC(25088, 4096) → BN1d → ReLU → CustomDropout
    FC(4096,  4096) → BN1d → ReLU → CustomDropout
    FC(4096,    37)
        ↓
Output: logits (B, 37)
```

## Task 2 — VGG11Localizer

```
VGG11 Encoder → Regression Head
    AdaptiveAvgPool → Flatten
    FC(25088, 1024) → ReLU
    FC(1024,   256) → ReLU
    FC(256,      4) → ReLU
        ↓
Output: [cx, cy, w, h] pixel coords (B, 4)
```

## Task 3 — VGG11UNet

```
VGG11 Encoder (contracting path)
        ↓
U-Net Decoder (expansive path)
    ConvTranspose2d + skip connection concatenation at each stage
    up5 → dec5 → up4 → dec4 → up3 → dec3 → up2 → dec2 → up1 → dec1
        ↓
Conv(1x1) → Output: logits (B, 3, 224, 224)
```

## Task 4 — MultiTaskPerceptionModel

```
Single shared VGG11 Encoder
        ↓
    ┌───────────────────────────────┐
    ↓               ↓              ↓
Cls Head       Loc Head       Seg Decoder
(B, 37)         (B, 4)     (B, 3, 224, 224)
```

---

# Installation

Clone the repository:

```bash
git clone <repo_link>
cd da6401_assignment_2
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Required libraries:

- torch
- numpy
- matplotlib
- scikit-learn
- wandb
- albumentations

---

# Training

## Train all tasks sequentially:

```bash
python train.py \
    --task all \
    --data_root ./data/oxford-iiit-pet \
    --epochs 20 \
    --batch_size 16 \
    --lr 1e-4
```

## Train individual tasks:

```bash
# Task 1 — Classification
python train.py --task 1 --data_root ./data/oxford-iiit-pet --epochs 20

# Task 2 — Localization
python train.py --task 2 --data_root ./data/oxford-iiit-pet --epochs 20

# Task 3 — Segmentation (with strategy)
python train.py --task 3 --data_root ./data/oxford-iiit-pet \
    --seg_strategy full   # frozen | partial | full

# Task 4 — Multi-task
python train.py --task 4 --data_root ./data/oxford-iiit-pet \
    --lambda_cls 1.0 --lambda_loc 1.0 --lambda_seg 1.0
```

---

# Training Parameters

| Argument         | Description                              | Default  |
| ---------------- | ---------------------------------------- | -------- |
| `--task`         | Task to train: `1`, `2`, `3`, `4`, `all` | `all`    |
| `--data_root`    | Path to Oxford-IIIT Pet dataset          | required |
| `--epochs`       | Number of training epochs                | `20`     |
| `--batch_size`   | Mini-batch size                          | `16`     |
| `--lr`           | Learning rate                            | `1e-4`   |
| `--dropout_p`    | Dropout probability in FC layers         | `0.5`    |
| `--seg_strategy` | Transfer learning strategy for Task 3    | `full`   |
| `--lambda_cls`   | Classification loss weight (Task 4)      | `1.0`    |
| `--lambda_loc`   | Localization loss weight (Task 4)        | `1.0`    |
| `--lambda_seg`   | Segmentation loss weight (Task 4)        | `1.0`    |
| `--num_workers`  | DataLoader workers                       | `4`      |

---

# Inference

Run the unified pipeline on a single image:

```bash
python inference.py \
    --image path/to/pet.jpg
```

Evaluate on the test set:

```bash
python inference.py \
    --eval_test \
    --data_root ./data/oxford-iiit-pet
```

---

# W&B Experiments

Login to W&B:

```bash
wandb login
```

---

# Evaluation Metrics

| Task           | Primary Metric  | Secondary Metric |
| -------------- | --------------- | ---------------- |
| Classification | Macro F1-Score  | Accuracy         |
| Localization   | Mean IoU        | -                |
| Segmentation   | Dice Score      | Pixel Accuracy   |
| Multi-task     | Macro F1 + Dice | All above        |

---

# Key Design Decisions

**Custom Dropout:**
Inverted dropout implemented via `torch.bernoulli`. Scaling by `1/(1-p)`
at train time ensures the inference pass is a pure identity with no
rescaling needed.

**BatchNorm Placement:**
BN placed after every Conv2d and before ReLU. Reduces internal covariate
shift, stabilises gradients, and allows higher stable learning rates.
Without BN, 86% of Block 3 activations were dead (ReLU killed) vs 51%
with BN.

**Custom IoU Loss:**
Converts `[cx, cy, w, h]` to corner format internally, computes
intersection and union areas, returns `1 - IoU`. Gradients flow through
the differentiable area computation.

**Segmentation Loss:**
Combined CrossEntropy + Dice Loss. CE provides stable per-pixel gradients;
Dice handles class imbalance in trimaps where background pixels dominate.
Dice Score (0.68) vs Pixel Accuracy (0.84) gap confirms the imbalance.

**Transfer Learning:**
Full fine-tuning of the VGG11 backbone was used for Tasks 2, 3, and 4.
Early blocks (edges, textures) converge quickly from the pretrained
initialisation; later blocks adapt toward task-specific spatial features.

---

# W&B Report Link:

https://wandb.ai/priyanshuvsp2841-indian-institute-of-technology-madras/da6401-assignment2/reports/DA6401_Assignment_2--VmlldzoxNjQ2MDY4MQ?accessToken=idp4svfs7ujzxm1nn1abegoqp3nhy8kjbwbmvscvmhf0zl6pc990waa62leua9eh

# GitHub Link:

https://github.com/priyanshu-verma87/da6401_assignment_2.git
