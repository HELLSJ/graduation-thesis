# Retinal Vessel Segmentation Project

## Project Overview

This project is a deep learning-based retinal vessel segmentation system that supports two model architectures: UNet and UNet++. Each architecture can use either EfficientNet or ResNet as the backbone. The system is primarily used for retinal vessel segmentation tasks in medical image processing, which can help doctors more accurately diagnose retinal-related diseases.

## Project Structure

```
graduation thesis/
├── src/                      # Source code folder
│   ├── __pycache__/         # Cache files
│   ├── results/             # Results directory
│   │   ├── evaluation/      # Evaluation results
│   │   ├── generalization_test/  # Generalization test results
│   │   ├── unet_efficientnet_train/  # UNet with EfficientNet training results
│   │   ├── unet_resnet_train/  # UNet with ResNet training results
│   │   ├── unetpp_efficientnet_train/  # UNet++ with EfficientNet training results
│   │   ├── unetpp_resnet_train/  # UNet++ with ResNet training results
│   │   └── comprehensive_evaluation_report.txt  # Comprehensive evaluation report
│   ├── callbacks.py         # Callback functions
│   ├── config.py            # Configuration parameters
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── evaluate_model_unet.py  # UNet evaluation script
│   ├── evaluate_model_unetpp.py  # UNet++ evaluation script
│   ├── requirements.txt     # Dependency list
│   ├── test_generalization_unet.py  # UNet generalization test script
│   ├── test_generalization_unetpp.py  # UNet++ generalization test script
│   ├── unet_model.py        # UNet model definition
│   ├── unet_train.py        # UNet training script
│   ├── unetpp_model.py      # UNet++ model definition
│   └── unetpp_train.py      # UNet++ training script
├── data/                   # Dataset folder
│   ├── CHASE_DB1/          # CHASE_DB1 dataset
│   ├── DRIVE/              # DRIVE dataset
│   ├── HRF/                # HRF dataset
│   └── STARE/              # STARE dataset
├── user-manual.md           # User manual (English)
├── user-manual-cn.md        # User manual (Chinese)
├── maintenance-manual.md    # Maintenance manual (English)
├── maintenance-manual-cn.md # Maintenance manual (Chinese)
└── README.md                # Project documentation
```

## Model Architecture

This project implements two model architectures, each supporting two backbones:

### 1. UNet Model
- **EfficientNet V2 B2 backbone**: Lightweight model, suitable for resource-constrained environments
- **ResNet50 backbone**: Deeper model, may achieve better performance

### 2. UNet++ Model
- **EfficientNet V2 B2 backbone**: Lightweight model, suitable for resource-constrained environments
- **ResNet50 backbone**: Deeper model, may achieve better performance

## Install Dependencies

First, install the necessary dependencies:

```bash
cd src
pip install -r requirements.txt
```

## Train Model

### Train UNet Model

- Using EfficientNet backbone (default):
  ```bash
  cd src
  python unet_train.py
  ```

- Using ResNet backbone:
  ```bash
  cd src
  python unet_train.py --backbone resnet
  ```

- Specify custom results directory:
  ```bash
  cd src
  python unet_train.py --backbone resnet --results-dir custom_results
  ```

### Train UNet++ Model

- Using EfficientNet backbone (default):
  ```bash
  cd src
  python unetpp_train.py
  ```

- Using ResNet backbone:
  ```bash
  cd src
  python unetpp_train.py --backbone resnet
  ```

- Specify custom results directory:
  ```bash
  cd src
  python unetpp_train.py --backbone resnet --results-dir custom_results
  ```

## Evaluate Model

### Evaluate UNet Model

- Evaluate EfficientNet backbone model (default):
  ```bash
  cd src
  python evaluate_model_unet.py
  ```

- Evaluate ResNet backbone model:
  ```bash
  cd src
  python evaluate_model_unet.py --backbone resnet
  ```

### Evaluate UNet++ Model

- Evaluate EfficientNet backbone model (default):
  ```bash
  cd src
  python evaluate_model_unetpp.py
  ```

- Evaluate ResNet backbone model:
  ```bash
  cd src
  python evaluate_model_unetpp.py --backbone resnet
  ```

## Test Generalization Ability

### Test UNet Model Generalization Ability

- Test EfficientNet backbone model (default):
  ```bash
  cd src
  python test_generalization_unet.py
  ```

- Test ResNet backbone model:
  ```bash
  cd src
  python test_generalization_unet.py --backbone resnet
  ```

### Test UNet++ Model Generalization Ability

- Test EfficientNet backbone model (default):
  ```bash
  cd src
  python test_generalization_unetpp.py
  ```

- Test ResNet backbone model:
  ```bash
  cd src
  python test_generalization_unetpp.py --backbone resnet
  ```

## Training Configuration

- **Dataset**: DRIVE retinal vessel segmentation dataset
- **Models**:
  - UNet with EfficientNet V2 B2 backbone
  - UNet with ResNet50 backbone
  - UNet++ with EfficientNet V2 B2 backbone
  - UNet++ with ResNet50 backbone
- **Image Size**: 512x512
- **Batch Size**: 6
- **Training Epochs**: 1001
- **Learning Rate**: 0.0001
- **Prediction Visualization**: Generate prediction images every 40 epochs

## Output Results

### Training Results

Training results are saved in the following directories:
- UNet with EfficientNet: `src/results/unet_efficientnet_train/`
- UNet with ResNet: `src/results/unet_resnet_train/`
- UNet++ with EfficientNet: `src/results/unetpp_efficientnet_train/`
- UNet++ with ResNet: `src/results/unetpp_resnet_train/`

Each directory contains:
- `best_model.keras`: Best model weights
- `training_summary.txt`: Training summary
- `training_record.json`: Detailed training record
- `loss_curve_large.png`: Loss curve
- `iou_curve_large.png`: IoU curve
- `dice_curve_large.png`: Dice curve
- `prediction_epoch_*.png`: Prediction results at different epochs

### Evaluation Results

Evaluation results are saved in the following directories:
- UNet with EfficientNet: `src/results/evaluation/unet/EfficientNet/`
- UNet with ResNet: `src/results/evaluation/unet/ResNet/`
- UNet++ with EfficientNet: `src/results/evaluation/unetpp/EfficientNet/`
- UNet++ with ResNet: `src/results/evaluation/unetpp/ResNet/`

Each directory contains:
- `evaluation_report.txt`: Evaluation report
- `evaluation_visualization_*.png`: Evaluation visualization results

### Generalization Test Results

Generalization test results are saved in the following directories:
- UNet with EfficientNet: `src/results/generalization_test/unet/EfficientNet/`
- UNet with ResNet: `src/results/generalization_test/unet/ResNet/`
- UNet++ with EfficientNet: `src/results/generalization_test/unetpp/EfficientNet/`
- UNet++ with ResNet: `src/results/generalization_test/unetpp/RESNET/`

Each directory contains:
- `generalization_summary.txt`: Generalization test summary
- `CHASE_DB1_evaluation_report.txt`: CHASE_DB1 dataset evaluation report
- `HRF_evaluation_report.txt`: HRF dataset evaluation report
- `*_visualization_*.png`: Visualization results

### Comprehensive Evaluation Report

The comprehensive evaluation report is saved at: `src/results/comprehensive_evaluation_report.txt`

This report contains performance comparisons of all models on the DRIVE validation set and generalization test sets.

## Configuration Parameters

You can modify training parameters in `src/config.py`:

```python
# Image and batch settings
IMAGE_SIZE = 512
BATCH_SIZE = 6

# Dataset paths
ROOT_DIR = 'd:\\graduation thesis\\data'
TRAIN_IMAGES_DIR = 'DRIVE/training/images'
TRAIN_MASKS_DIR = 'DRIVE/training/1st_manual'

# Model parameters
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
N_CLASSES = 1
ACTIVATION = 'sigmoid'

# Training parameters
EPOCHS = 1001
LEARNING_RATE = 0.0001

ENCODER_NAME = 'efficientnet_v2_b2'
# Backbone model
ENCODER_NAME_REFIX = 'resnet50'
ENCODER_DEPTH = 4
```

## Dataset Paths

Ensure the dataset is placed in the correct location:

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

## Notes

- Result directories are automatically created during training
- Prediction images are saved every 40 epochs
- The best model is automatically saved based on validation set dice score
- Early stopping mechanism is used, training will automatically stop if no improvement for 200 epochs
- UNet++ model contains more network layers, has larger parameters, and takes longer to train
- Generalization test evaluates model performance on CHASE_DB1 and HRF datasets

## Documentation

- **User Manual**: `user-manual.md` - Detailed usage instructions (English)
- **User Manual (Chinese)**: `user-manual-cn.md` - Detailed usage instructions (Chinese)
- **Maintenance Manual**: `maintenance-manual.md` - System implementation details and maintenance guide (English)
- **Maintenance Manual (Chinese)**: `maintenance-manual-cn.md` - System implementation details and maintenance guide (Chinese)