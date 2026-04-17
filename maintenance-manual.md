# Retinal Vessel Segmentation Model Maintenance Manual

## 1. System Overview

This system is a deep learning-based retinal vessel segmentation model that supports two model architectures: UNet and UNet++. Each architecture can use either EfficientNet or ResNet as the backbone. The system is primarily used for retinal vessel segmentation tasks in medical image processing.

## 2. System Installation

### 2.1 Environment Preparation

1. **Python Environment**: Ensure Python 3.7 or higher is installed
2. **Virtual Environment**: It is recommended to use a virtual environment to isolate dependencies
3. **Dependency Libraries**: Install the required Python libraries

### 2.2 Installation Steps

1. **Clone the Project**:
   ```bash
   git clone <project_url>
   cd graduation thesis
   ```

2. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   # Windows activation
   venv\Scripts\activate
   # Linux/Mac activation
   # source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   cd src
   pip install -r requirements.txt
   ```

4. **Prepare Dataset**:
   Ensure the dataset directory structure is correct, see Section 3.4 of the user manual

## 3. System Build

### 3.1 Code Compilation

This project is written in Python and does not require compilation steps. Simply run the corresponding Python scripts directly.

### 3.2 Build Process

1. **Data Preparation**: Ensure the dataset is correctly placed in the `data` directory
2. **Model Training**: Run the training script to train the model
3. **Model Evaluation**: Run the evaluation script to evaluate model performance
4. **Generalization Test**: Run the generalization test script to test the model's performance on different datasets

## 4. Hardware/Software Dependencies

### 4.1 Hardware Dependencies

- **CPU**: Intel Core i5 or higher
- **Memory**: 8GB or more
- **Storage**: At least 10GB
- **GPU (Optional)**: CUDA-enabled NVIDIA GPU for accelerated training

### 4.2 Software Dependencies

| Dependency | Version Requirement | Purpose |
|------------|---------------------|--------|
| tensorflow | 2.20.0 | Deep learning framework |
| keras | 3.12.0 | High-level neural network API |
| medicai | 0.0.3 | Medical image processing library |
| pillow | 10.4.0 | Image processing |
| numpy | 1.26.4 | Numerical computation |
| matplotlib | 3.9.2 | Data visualization |
| pandas | 2.2.2 | Data processing |

## 5. System File Organization

### 5.1 Directory Structure

```
graduation thesis/
├── src/                 # Source code directory
│   ├── __pycache__/     # Cache files
│   ├── results/         # Results directory
│   │   ├── evaluation/  # Evaluation results
│   │   ├── generalization_test/  # Generalization test results
│   │   ├── unet_efficientnet_train/  # UNet with EfficientNet training results
│   │   ├── unet_resnet_train/  # UNet with ResNet training results
│   │   ├── unetpp_efficientnet_train/  # UNet++ with EfficientNet training results
│   │   ├── unetpp_resnet_train/  # UNet++ with ResNet training results
│   │   └── comprehensive_evaluation_report.txt  # Comprehensive evaluation report
│   ├── callbacks.py     # Callback functions
│   ├── config.py        # Configuration file
│   ├── data_loader.py   # Data loading
│   ├── evaluate_model_unet.py  # UNet evaluation script
│   ├── evaluate_model_unetpp.py  # UNet++ evaluation script
│   ├── requirements.txt  # Dependencies
│   ├── test_generalization_unet.py  # UNet generalization test script
│   ├── test_generalization_unetpp.py  # UNet++ generalization test script
│   ├── unet_model.py    # UNet model definition
│   ├── unet_train.py    # UNet training script
│   ├── unetpp_model.py  # UNet++ model definition
│   └── unetpp_train.py  # UNet++ training script
├── data/                # Dataset directory
│   ├── CHASE_DB1/       # CHASE_DB1 dataset
│   ├── DRIVE/           # DRIVE dataset
│   └── HRF/             # HRF dataset
├── .venv/               # Virtual environment
├── user-manual-cn.md    # Chinese user manual
└── maintenance-manual-cn.md  # Chinese maintenance manual
```

### 5.2 Key File Description

| File | Function | Location |
|------|----------|----------|
| config.py | Configuration parameters | src/config.py |
| unet_model.py | UNet model definition | src/unet_model.py |
| unetpp_model.py | UNet++ model definition | src/unetpp_model.py |
| unet_train.py | UNet training script | src/unet_train.py |
| unetpp_train.py | UNet++ training script | src/unetpp_train.py |
| evaluate_model_unet.py | UNet evaluation script | src/evaluate_model_unet.py |
| evaluate_model_unetpp.py | UNet++ evaluation script | src/evaluate_model_unetpp.py |
| test_generalization_unet.py | UNet generalization test script | src/test_generalization_unet.py |
| test_generalization_unetpp.py | UNet++ generalization test script | src/test_generalization_unetpp.py |
| data_loader.py | Data loading module | src/data_loader.py |
| callbacks.py | Training callback functions | src/callbacks.py |

## 6. Space and Memory Requirements

### 6.1 Storage Space

- **Dataset**: Approximately 5GB
- **Model Files**: Each model is approximately 100-300MB
- **Training Results**: Each model is approximately 500MB (including prediction results and curves)
- **Total Storage**: At least 10GB

### 6.2 Memory Requirements

- **Training**: At least 8GB of memory is recommended, 16GB or more is better
- **Evaluation and Testing**: At least 4GB of memory

## 7. Source Code File List

| File Name | Function | Description |
|-----------|----------|-------------|
| config.py | Configuration file | Defines image size, batch size, learning rate, etc. |
| unet_model.py | UNet model definition | Defines UNet model architecture, supports EfficientNet and ResNet backbones |
| unetpp_model.py | UNet++ model definition | Defines UNet++ model architecture, supports EfficientNet and ResNet backbones |
| unet_train.py | UNet training script | Trains UNet model, supports command line parameter to select backbone |
| unetpp_train.py | UNet++ training script | Trains UNet++ model, supports command line parameter to select backbone |
| evaluate_model_unet.py | UNet evaluation script | Evaluates UNet model performance, generates evaluation report and visualization results |
| evaluate_model_unetpp.py | UNet++ evaluation script | Evaluates UNet++ model performance, generates evaluation report and visualization results |
| test_generalization_unet.py | UNet generalization test script | Tests UNet model performance on CHASE_DB1 and HRF datasets |
| test_generalization_unetpp.py | UNet++ generalization test script | Tests UNet++ model performance on CHASE_DB1 and HRF datasets |
| data_loader.py | Data loading module | Loads and preprocesses datasets |
| callbacks.py | Callback functions | Defines callback functions during training, such as displaying prediction results |

## 8. Key Constants

| Constant | Value | Description | Location |
|----------|-------|-------------|----------|
| IMAGE_SIZE | 512 | Image size | src/config.py:4 |
| BATCH_SIZE | 6 | Batch size | src/config.py:5 |
| ROOT_DIR | 'data' | Data root directory | src/config.py:8 |
| EPOCHS | 1001 | Training epochs | src/config.py:18 |
| LEARNING_RATE | 0.0001 | Learning rate | src/config.py:19 |
| ENCODER_NAME | 'efficientnet_v2_b2' | EfficientNet encoder name | src/config.py:21 |
| ENCODER_NAME_REFIX | 'resnet50' | ResNet encoder name | src/config.py:23 |
| ENCODER_DEPTH | 4 | Encoder depth | src/config.py:24 |

## 9. Main Classes and Methods

### 9.1 UNet Model

| Method | Description | Parameters | Return Value | Location |
|--------|-------------|------------|--------------|----------|
| build_model | Build UNet model | backbone: str - Select backbone ('efficientnet' or 'resnet') | keras.Model - Built model | src/unet_model.py:16 |
| compile_model | Compile UNet model | model: keras.Model - Model to compile<br>learning_rate: float - Learning rate | keras.Model - Compiled model | src/unet_model.py:51 |

### 9.2 UNet++ Model

| Method | Description | Parameters | Return Value | Location |
|--------|-------------|------------|--------------|----------|
| build_improved_unet | Build UNet++ model | backbone: str - Select backbone ('efficientnet' or 'resnet') | keras.Model - Built model | src/unetpp_model.py:17 |
| compile_improved_model | Compile UNet++ model | model: keras.Model - Model to compile<br>learning_rate: float - Learning rate | keras.Model - Compiled model | src/unetpp_model.py:54 |

### 9.3 Training Scripts

| Method | Description | Parameters | Return Value | Location |
|--------|-------------|------------|--------------|----------|
| parse_args | Parse command line arguments | None | argparse.Namespace - Parsed arguments | src/unet_train.py:37<br>src/unetpp_train.py:37 |
| main | Training main function | None | None | src/unet_train.py:48<br>src/unetpp_train.py:48 |
| save_training_record | Save training record | history: keras.callbacks.History - Training history<br>model: keras.Model - Trained model<br>training_time: float - Training time<br>save_dir: str - Save directory<br>backbone: str - Used backbone | dict - Training record | src/unet_train.py:123<br>src/unetpp_train.py:126 |
| plot_training_curves | Plot training curves | history: keras.callbacks.History - Training history<br>save_dir: str - Save directory<br>model: keras.Model - Trained model<br>training_time: float - Training time<br>backbone: str - Used backbone | None | src/unet_train.py:234<br>src/unetpp_train.py:234 |

### 9.4 Evaluation Scripts

| Method | Description | Parameters | Return Value | Location |
|--------|-------------|------------|--------------|----------|
| parse_args | Parse command line arguments | None | argparse.Namespace - Parsed arguments | src/evaluate_model_unet.py:136 |
| load_trained_model | Load trained model | backbone: str - Used backbone<br>model_dir: str - Model directory | keras.Model - Loaded model | src/evaluate_model_unet.py:149 |
| visualize_results | Visualize evaluation results | model: keras.Model - Evaluated model<br>dataset: tf.data.Dataset - Test dataset<br>save_dir: str - Save directory | None | src/evaluate_model_unet.py:202 |
| evaluate_model | Evaluate model | None | tuple - Evaluation metrics (loss, iou, dice, recall, precision) | src/evaluate_model_unet.py:300 |
| save_evaluation_results | Save evaluation results | loss: float - Loss value<br>iou: float - IoU value<br>dice: float - Dice value<br>recall: float - Recall<br>precision: float - Precision<br>save_dir: str - Save directory<br>backbone: str - Used backbone | None | src/evaluate_model_unet.py:353 |

### 9.5 Generalization Test Scripts

| Method | Description | Parameters | Return Value | Location |
|--------|-------------|------------|--------------|----------|
| parse_args | Parse command line arguments | None | argparse.Namespace - Parsed arguments | src/test_generalization_unet.py:214 |
| get_chase_db1_dataset | Load CHASE_DB1 dataset | None | tf.data.Dataset - CHASE_DB1 dataset | src/test_generalization_unet.py:139 |
| get_hrf_dataset | Load HRF dataset | None | tf.data.Dataset - HRF dataset | src/test_generalization_unet.py:168 |
| load_trained_model | Load trained model | backbone: str - Used backbone<br>model_dir: str - Model directory | keras.Model - Loaded model | src/test_generalization_unet.py:227 |
| visualize_results | Visualize test results | model: keras.Model - Tested model<br>dataset: tf.data.Dataset - Test dataset<br>save_dir: str - Save directory<br>dataset_name: str - Dataset name<br>backbone: str - Used backbone | None | src/test_generalization_unet.py:278 |
| test_on_dataset | Test on specific dataset | dataset_name: str - Dataset name<br>dataset: tf.data.Dataset - Test dataset<br>save_dir: str - Save directory<br>backbone: str - Used backbone | tuple - Test metrics (loss, dice, recall, precision) | src/test_generalization_unet.py:376 |
| test_generalization | Test model generalization ability | None | None | src/test_generalization_unet.py:427 |
| save_generalization_results | Save generalization test results | chase_loss: float - CHASE_DB1 loss value<br>chase_dice: float - CHASE_DB1 Dice value<br>chase_recall: float - CHASE_DB1 recall<br>chase_precision: float - CHASE_DB1 precision<br>hrf_loss: float - HRF loss value<br>hrf_dice: float - HRF Dice value<br>hrf_recall: float - HRF recall<br>hrf_precision: float - HRF precision<br>save_dir: str - Save directory<br>backbone: str - Used backbone | None | src/test_generalization_unet.py:469 |

### 9.6 Command Line Arguments

#### 9.6.1 Training Scripts (unet_train.py, unetpp_train.py)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --backbone | str | 'efficientnet' | Backbone model to use ('efficientnet' or 'resnet') |

**Usage Examples:**
```bash
# Train UNet with EfficientNet backbone (default)
cd src && python unet_train.py

# Train UNet with ResNet backbone
cd src && python unet_train.py --backbone resnet

# Train UNet++ with EfficientNet backbone (default)
cd src && python unetpp_train.py

# Train UNet++ with ResNet backbone
cd src && python unetpp_train.py --backbone resnet
```

#### 9.6.2 Evaluation Scripts (evaluate_model_unet.py, evaluate_model_unetpp.py)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --backbone | str | 'efficientnet' | Backbone model to use ('efficientnet' or 'resnet') |
| --results-dir | str | None | Directory to save results (auto-generated if not specified) |
| --model-dir | str | None | Directory where the trained model is saved (auto-generated if not specified) |

**Usage Examples:**
```bash
# Evaluate UNet with EfficientNet backbone (default)
cd src && python evaluate_model_unet.py

# Evaluate UNet with ResNet backbone
cd src && python evaluate_model_unet.py --backbone resnet

# Specify custom results directory
cd src && python evaluate_model_unet.py --backbone resnet --results-dir custom_results

# Specify custom model directory
cd src && python evaluate_model_unet.py --backbone resnet --model-dir path/to/model
```

Note: The evaluation script needs to read the best_model saved after training. If you run the evaluation script without training first, you will see the following error:
![[eval error.png]]

#### 9.6.3 Generalization Test Scripts (test_generalization_unet.py, test_generalization_unetpp.py)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --backbone | str | 'efficientnet' | Backbone model to use ('efficientnet' or 'resnet') |
| --results-dir | str | None | Directory to save results (auto-generated if not specified) |
| --model-dir | str | None | Directory where the trained model is saved (auto-generated if not specified) |

**Usage Examples:**
```bash
# Test UNet++ with EfficientNet backbone (default)
cd src && python test_generalization_unetpp.py

# Test UNet++ with ResNet backbone
cd src && python test_generalization_unetpp.py --backbone resnet

# Specify custom results directory
cd src && python test_generalization_unetpp.py --backbone resnet --results-dir custom_results

# Specify custom model directory
cd src && python test_generalization_unetpp.py --backbone resnet --model-dir path/to/model
```

**Default Paths:**

When `--results-dir` is not specified:
- UNet with EfficientNet: `results/generalization_test/unet/EfficientNet`
- UNet with ResNet: `results/generalization_test/unet/ResNet`
- UNet++ with EfficientNet: `results/generalization_test/unetpp/EfficientNet`
- UNet++ with ResNet: `results/generalization_test/unetpp/ResNet`

When `--model-dir` is not specified:
- UNet with EfficientNet: `results/unet_efficientnet_train`
- UNet with ResNet: `results/unet_resnet_train`
- UNet++ with EfficientNet: `results/unetpp_efficientnet_train`
- UNet++ with ResNet: `results/unetpp_resnet_train`

Note: The generalization test script needs to read the best_model saved after training. If you run the test script without training first, you will see the following error:
![[test_error.png]]

## 10. File Paths

| Path | Description | Purpose |
|------|-------------|--------|
| data | Data root directory | Stores all datasets |
| data/DRIVE | DRIVE dataset | Used for training and validation |
| data/CHASE_DB1 | CHASE_DB1 dataset | Used for generalization test |
| data/HRF | HRF dataset | Used for generalization test |
| src/results | Results directory | Stores all training and testing results |
| src/results/unet_resnet_train | UNet with ResNet training results | Stores trained model and curves |
| src/results/evaluation/unet/ResNet | UNet with ResNet evaluation results | Stores evaluation reports and visualizations |
| src/results/generalization_test/unet/ResNet | UNet with ResNet generalization test results | Stores generalization test reports and visualizations |

## 11. Future Improvement Directions

### 11.1 Model Improvement

- **Model Architecture**: Explore more advanced segmentation models, such as Attention UNet, Nested UNet, etc.
- **Backbone**: Try other pre-trained models as backbones, such as Vision Transformer
- **Loss Function**: Explore loss functions more suitable for vessel segmentation, such as Focal Loss, Tversky Loss, etc.
- **Data Augmentation**: Add more data augmentation methods, such as rotation, scaling, contrast adjustment, etc.

### 11.2 Function Expansion

- **Real-time Segmentation**: Optimize model inference speed to achieve real-time segmentation
- **Multi-modal Fusion**: Fusion of different modal image information to improve segmentation performance
- **Automatic Disease Diagnosis**: Develop automatic disease diagnosis function based on segmentation results
- **User Interface**: Develop a graphical user interface for easy use by doctors

### 11.3 Performance Optimization

- **Model Compression**: Use knowledge distillation, quantization and other techniques to compress the model
- **Hardware Acceleration**: Optimize the model to fully utilize GPU or TPU acceleration
- **Parallel Computing**: Use multi-GPU parallel training to accelerate the model training process

## 12. Error Reporting

### 12.1 Common Errors

| Error Type | Error Message | Cause | Solution |
|------------|---------------|-------|----------|
| Model file not found | "Error: Model file not found" | Corresponding model not trained | Run the training script to train the model first |
| Out of memory | "OOM error" | Insufficient memory | Reduce batch size or image size |
| Import error | "ImportError: cannot import name" | Dependencies not installed correctly | Reinstall dependencies |
| Dataset path error | "FileNotFoundError" | Incorrect dataset path | Check dataset path and directory structure |
| Training crash | "CUDA out of memory" | Insufficient GPU memory | Reduce batch size or use CPU training |

### 12.2 Error Handling

- **Log Recording**: The system outputs detailed error information in the console
- **Exception Capture**: Key operations have exception capture to ensure the system does not crash
- **Error Prompt**: Error messages are clear and provide solution suggestions

### 12.3 Reporting Errors

If you encounter an error not listed, please record the following information and contact the author:
- Error message and stack trace
- Running environment (operating system, Python version, dependency library versions)
- Reproduction steps
- Expected behavior and actual behavior

## 13. Code Maintenance

### 13.1 Code Style

- Follow PEP 8 code style guidelines
- Clear and concise function and variable naming
- Appropriate comments in the code
- Modular design for easy maintenance and expansion

### 13.2 Version Control

- Use Git for version control
- Clear and concise commit messages
- Regularly update code and documentation

### 13.3 Testing

- After each code modification, run training and evaluation scripts to ensure normal functionality
- Test model performance with different backbones
- Test model generalization ability on different datasets

## 14. Summary

This maintenance manual provides detailed implementation information of the retinal vessel segmentation model, including system installation, file organization, code structure, key components, etc. Through this manual, developers can understand the working principle of the system, perform system maintenance and expansion.

The system adopts a modular design with clear code structure, which is easy to maintain and expand. In the future, system performance can be further improved by exploring new model architectures, optimizing training strategies, and adding functions.