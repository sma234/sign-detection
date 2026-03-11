# Traffic Sign Detection

Real-time traffic sign detection pipeline built for NVIDIA Jetson. The system uses HSV-based color segmentation to isolate sign candidates, then classifies them through template matching. A lightweight CNN (`SmallSignNet`) is included for training-based classification as an alternative backend.

## Overview

| Stage | Method |
|---|---|
| Candidate detection | HSV color segmentation + circularity filter |
| Classification | Normalized cross-correlation template matching |
| Alt. classification | Custom CNN (trained with PyTorch) |

**Supported sign classes:** STOP, AHEAD, RIGHT\_BLOCK, LEFT\_BLOCK, PARKING, ROUNDABOUT, HIGHWAY\_START, HIGHWAY\_END, PRIORITY\_ROAD, NO\_ENTRY

## Project Structure

```
.
├── jetson_main.py    # Main detection loop (template matching)
├── classify.py       # CNN-based classifier
├── model.py          # SmallSignNet architecture
├── train.py          # Training script
├── dataset.py        # PyTorch dataset loader
├── classes.py        # Shared class list
├── hsv_picker.py     # Utility for tuning HSV thresholds
├── templates/        # Reference images per class (for template matching)
├── dataset/          # Training images per class (for CNN)
└── traffic_signs.pth # Saved model weights
```

## Requirements

```
opencv-python
torch
torchvision
Pillow
numpy
```

Install with:

```bash
pip install opencv-python torch torchvision Pillow numpy
```

## Usage

### Run the detector

```bash
python jetson_main.py
```

Opens the default camera. Detected signs are drawn with a bounding box and label. Press `q` to quit.

### Train the CNN

Organize training images under `dataset/<CLASS_NAME>/` then run:

```bash
python train.py
```

Trains for 15 epochs and saves weights to `traffic_signs.pth`.

### Tune HSV thresholds

```bash
python hsv_picker.py
```

Click anywhere on the live frame to print the HSV value at that pixel.

## How It Works

1. **Color segmentation** — Each frame is blurred and converted to HSV. Masks are computed for red (two ranges, since red wraps around 0°), blue, green, and yellow.
2. **Morphological cleanup** — Opening removes noise; closing fills gaps in the mask.
3. **Contour filtering** — Contours smaller than 300 px² or with circularity below 0.5 are discarded.
4. **Template matching** — Each surviving ROI is compared against reference images using `TM_CCOEFF_NORMED`. A match is accepted when the score exceeds 0.4.

## Model Architecture

`SmallSignNet` is a compact CNN designed to run efficiently on edge hardware:

```
Input (3 × 32 × 32)
  → Conv2d(3, 16, 3) → ReLU → MaxPool  →  16 × 16
  → Conv2d(16, 32, 3) → ReLU → MaxPool →   8 × 8
  → Flatten → Linear(2048, 64) → ReLU
  → Linear(64, num_classes)
```

Trained with Adam (lr=1e-3) and cross-entropy loss over 15 epochs.

## Hardware

Developed and tested on an **NVIDIA Jetson** board. Runs on any CUDA-capable GPU or CPU fallback.
