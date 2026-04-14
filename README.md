# 🩸 Retinal Vessel Segmentation Project

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.12.0-red?logo=keras&logoColor=white)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/HELLSJ/graduation-thesis?style=social)](https://github.com/HELLSJ/graduation-thesis)

**A Deep Learning-Based Retinal Vessel Segmentation System**

[English](#overview) | [中文文档](README-cn.md)

</div>

---

## 📋 Overview

This project is a deep learning-based retinal vessel segmentation system that supports **UNet** and **UNet++** architectures with **EfficientNet** or **ResNet** backbones. The system is designed for medical image processing tasks, specifically for segmenting retinal blood vessels, which can assist doctors in diagnosing retinal-related diseases more accurately.

### ✨ Key Features

- 🔬 **Multiple Model Architectures**: UNet and UNet++ with different backbones
- 🎯 **High Performance**: UNet++ with EfficientNet achieves 0.6539 Dice score on DRIVE validation set
- 📊 **Comprehensive Evaluation**: Training, validation, and generalization testing on multiple datasets
- 🖼️ **Visualization**: Automatic generation of training curves and prediction results
- 📁 **Organized Output**: Structured result directories with detailed reports

---

## 🏗️ Model Architecture

### Supported Models

| Model | Backbone | Parameters | Description |
|-------|----------|------------|-------------|
| **UNet** | EfficientNet V2 B2 | ~9.2M | Lightweight model with good balance |
| **UNet** | ResNet50 | ~32.5M | Deeper model with rich features |
| **UNet++** | EfficientNet V2 B2 | ~9.2M | Advanced architecture with nested skip pathways |
| **UNet++** | ResNet50 | ~32.5M | Most powerful combination for complex segmentation |

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Image (512x512x3)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Encoder (Backbone)                        │
│  ┌─────────────────┐        ┌─────────────────┐            │
│  │ EfficientNet    │   OR   │ ResNet50        │            │
│  │ V2 B2           │        │                 │            │
│  └─────────────────┘        └─────────────────┘            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Decoder with Skip Connections             │
│         (UNet or UNet++ Architecture)                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output Mask (512x512x1)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
graduation thesis/
├── src/                              # Source code
│   ├── results/                      # Training and evaluation results
│   │   ├── evaluation/               # Evaluation results
│   │   ├── generalization_test/      # Generalization test results
│   │   ├── unet_efficientnet_train/  # UNet + EfficientNet results
│   │   ├── unet_resnet_train/        # UNet + ResNet results
│   │   ├── unetpp_efficientnet_train/ # UNet++ + EfficientNet results
│   │   └── unetpp_resnet_train/      # UNet++ + ResNet results
│   ├── callbacks.py                  # Training callbacks
│   ├── config.py                     # Configuration parameters
│   ├── data_loader.py                # Data loading utilities
│   ├── evaluate_model_unet.py        # UNet evaluation script
│   ├── evaluate_model_unetpp.py      # UNet++ evaluation script
│   ├── test_generalization_unet.py   # UNet generalization test
│   ├── test_generalization_unetpp.py # UNet++ generalization test
│   ├── unet_model.py                 # UNet model definition
│   ├── unet_train.py                 # UNet training script
│   ├── unetpp_model.py               # UNet++ model definition
│   └── unetpp_train.py               # UNet++ training script
├── data/                             # Dataset directory
│   ├── CHASE_DB1/                    # CHASE_DB1 dataset
│   ├── DRIVE/                        # DRIVE dataset
│   └── HRF/                          # HRF dataset
├── user-manual.md                    # User manual (English)
├── user-manual-cn.md                 # User manual (Chinese)
├── maintenance-manual.md             # Maintenance manual (English)
├── maintenance-manual-cn.md          # Maintenance manual (Chinese)
└── README.md                         # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- TensorFlow 2.20.0
- CUDA-enabled GPU (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/HELLSJ/graduation-thesis.git
   cd graduation-thesis
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   cd src
   pip install -r requirements.txt
   ```

### Training

#### UNet Model

```bash
# Using EfficientNet backbone (default)
python unet_train.py

# Using ResNet backbone
python unet_train.py --backbone resnet

# Custom results directory
python unet_train.py --backbone resnet --results-dir custom_results
```

#### UNet++ Model

```bash
# Using EfficientNet backbone (default)
python unetpp_train.py

# Using ResNet backbone
python unetpp_train.py --backbone resnet
```

### Evaluation

```bash
# Evaluate UNet model
python evaluate_model_unet.py --backbone resnet

# Evaluate UNet++ model
python evaluate_model_unetpp.py --backbone resnet
```

### Generalization Testing

```bash
# Test UNet model generalization
python test_generalization_unet.py --backbone resnet

# Test UNet++ model generalization
python test_generalization_unetpp.py --backbone resnet
```

---

## 📊 Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Dataset** | DRIVE | Retinal vessel segmentation dataset |
| **Image Size** | 512×512 | Input image dimensions |
| **Batch Size** | 6 | Training batch size |
| **Epochs** | 1001 | Maximum training epochs |
| **Learning Rate** | 0.0001 | Adam optimizer learning rate |
| **Early Stopping** | 200 epochs | Stop if no improvement |
| **Prediction Interval** | 40 epochs | Save predictions every N epochs |

---

## 📈 Training Outputs

Each training session generates the following files:

| File | Description |
|------|-------------|
| `best_model.keras` | Best model weights based on validation Dice score |
| `training_summary.txt` | Human-readable training summary |
| `training_record.json` | Detailed training metrics in JSON format |
| `loss_curve_large.png` | Training and validation loss curves |
| `iou_curve_large.png` | Training and validation IoU curves |
| `dice_curve_large.png` | Training and validation Dice curves |
| `prediction_epoch_*.png` | Prediction results at different epochs |

---

## ⚙️ Configuration

Edit `src/config.py` to customize training parameters:

```python
# Image and batch settings
IMAGE_SIZE = 512
BATCH_SIZE = 6

# Dataset paths
ROOT_DIR = 'data'
TRAIN_IMAGES_DIR = 'DRIVE/training/images'
TRAIN_MASKS_DIR = 'DRIVE/training/1st_manual'

# Model parameters
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
N_CLASSES = 1
ACTIVATION = 'sigmoid'

# Training parameters
EPOCHS = 1001
LEARNING_RATE = 0.0001

# Backbone options
ENCODER_NAME = 'efficientnet_v2_b2'
ENCODER_NAME_REFIX = 'resnet50'
ENCODER_DEPTH = 4
```

---

## 📦 Dataset Structure

Ensure your dataset is organized as follows:

```
data/
├── CHASE_DB1/
│   ├── Images/
│   └── Masks/
├── DRIVE/
│   ├── test/
│   │   ├── images/
│   │   └── mask/
│   └── training/
│       ├── 1st_manual/
│       ├── images/
│       └── mask/
└── HRF/
    ├── images/
    ├── manual1/
    └── mask/
```

---

## 📝 Notes

- ✅ Result directories are automatically created during training
- ✅ Prediction images are saved every 40 epochs
- ✅ Best model is automatically saved based on validation Dice score
- ✅ Early stopping mechanism prevents overfitting
- ⚠️ UNet++ has more parameters and requires longer training time
- ⚠️ Generalization testing evaluates performance on unseen datasets

---

## 📚 Documentation

| Document | Language | Description |
|----------|----------|-------------|
| [User Manual](user-manual.md) | English | Detailed usage instructions |
| [User Manual](user-manual-cn.md) | Chinese | 详细使用说明 |
| [Maintenance Manual](maintenance-manual.md) | English | System implementation details |
| [Maintenance Manual](maintenance-manual-cn.md) | Chinese | 系统实现细节和维护指南 |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 🙏 Acknowledgments

- [DRIVE Dataset](https://drive.grand-challenge.org/) - Digital Retinal Images for Vessel Extraction
- [CHASE_DB1 Dataset](https://datasetninja.com/chase-db1) - Child Heart and Health Study in England
- [HRF Dataset](https://www5.cs.fau.de/research/data/fundus-images/) - High-Resolution Fundus Image Database

---

<div align="center">

**Made with ❤️ for Medical Image Analysis**

[⬆ Back to Top](#-retinal-vessel-segmentation-project)

</div>
