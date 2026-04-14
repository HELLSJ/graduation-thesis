# Retinal Vessel Segmentation Model User Manual

## 1. System Overview

This system is a deep learning-based retinal vessel segmentation model that supports two model architectures: UNet and UNet++. Each architecture can use either EfficientNet or ResNet as the backbone. The system is primarily used for retinal vessel segmentation tasks in medical image processing, which can help doctors more accurately diagnose retinal-related diseases.

### 1.1 Main Functions

- Train retinal vessel segmentation models
- Evaluate model performance
- Test model generalization ability
- Generate visualization results and evaluation reports

### 1.2 System Architecture

The system consists of the following main components:
- Data loading module: Responsible for loading and preprocessing datasets
- Model definition module: Defines UNet and UNet++ model architectures
- Training module: Trains the model and saves the best model
- Evaluation module: Evaluates model performance and generates reports
- Generalization test module: Tests model performance on different datasets

## 2. Environment Requirements

### 2.1 Hardware Requirements

- CPU: Intel Core i5 or higher
- Memory: 8GB or more
- Storage: At least 10GB
- GPU (Optional): CUDA-enabled NVIDIA GPU for accelerated training

### 2.2 Software Requirements

- Operating System: Windows 10/11
- Python: 3.7 or higher version
- Dependencies:
  - tensorflow
  - keras
  - medicai
  - pillow
  - numpy
  - matplotlib
  - pandas

## 3. Installation Guide

### 3.1 Clone the Project

Clone the project to your local directory:

```bash
git clone https://github.com/HELLSJ/graduation-thesis.git
cd graduation-thesis
```
![[git-clone.png]]
### 3.2 Create Virtual Environment

```bash
# Create virtual environment using conda
conda create -n retinal-segmentation python=3.9
conda activate retinal-segmentation

# Or create virtual environment using venv
python -m venv venv
# Windows activation
venv\Scripts\activate
# Linux/Mac activation
# source venv/bin/activate
```

### 3.3 Install Dependencies

```bash
cd src
pip install -r requirements.txt
```
![[environment_create.png]]
### 3.4 Prepare Dataset

The system uses the following datasets:
- DRIVE: Used for training and validation
- CHASE_DB1: Used for generalization test
- HRF: Used for generalization test

Please ensure these datasets have been downloaded and placed in the `data` directory with the following structure:

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

## 4. Usage Guide

### 4.1 Train Model

#### 4.1.1 Train UNet Model

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

#### 4.1.2 Train UNet++ Model

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

### 4.2 Evaluate Model

#### 4.2.1 Evaluate UNet Model

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

- Specify custom results directory:

```bash
cd src
python evaluate_model_unet.py --backbone resnet --results-dir custom_results
```

- Specify custom model directory:

```bash
cd src
python evaluate_model_unet.py --backbone resnet --model-dir path/to/model
```

#### 4.2.2 Evaluate UNet++ Model

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

### 4.3 Test Generalization Ability

#### 4.3.1 Test UNet Model Generalization Ability

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

#### 4.3.2 Test UNet++ Model Generalization Ability

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

## 5. Results View

### 5.1 Training Results

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

### 5.2 Evaluation Results

Evaluation results are saved in the following directories:
- UNet with EfficientNet: `src/results/evaluation/unet/EfficientNet/`
- UNet with ResNet: `src/results/evaluation/unet/ResNet/`
- UNet++ with EfficientNet: `src/results/evaluation/unetpp/EfficientNet/`
- UNet++ with ResNet: `src/results/evaluation/unetpp/ResNet/`

Each directory contains:
- `evaluation_report.txt`: Evaluation report
- `evaluation_visualization_*.png`: Evaluation visualization results

### 5.3 Generalization Test Results

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

### 5.4 Comprehensive Evaluation Report

The comprehensive evaluation report is saved at: `src/results/comprehensive_evaluation_report.txt`

This report contains performance comparisons of all models on the DRIVE validation set and generalization test sets.

## 6. Common Issues and Solutions

### 6.1 Model File Not Found

**Issue**: Running the evaluation script prompts "Model file not found"

**Solution**: Ensure the corresponding model has been trained. Training commands are as follows:
- Train UNet with ResNet: `python unet_train.py --backbone resnet`
- Train UNet++ with ResNet: `python unetpp_train.py --backbone resnet`

### 6.2 Out of Memory

**Issue**: Memory不足错误 occurs during training

**Solution**:
- Reduce batch size (modify BATCH_SIZE in config.py)
- Use GPU for training
- Reduce image size (modify IMAGE_SIZE in config.py)

### 6.3 Import Error

**Issue**: Import error occurs when running the script

**Solution**: Ensure all dependencies are installed correctly. Run:
```bash
pip install -r requirements.txt
```

## 7. Example Run

### 7.1 Train UNet with ResNet Model

```bash
cd src
python unet_train.py --backbone resnet
```
![[unet_resnet_train.png]]
### 7.2 Evaluate UNet with ResNet Model

```bash
cd src
python evaluate_model_unet.py --backbone resnet
```
![[unet_resnet_eval.png]]
### 7.3 Test UNet with ResNet Model Generalization Ability

```bash
cd src
python test_generalization_unet.py --backbone resnet
```
![[unet_resnet_test.png]]