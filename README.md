# 视网膜血管分割项目

## 项目概述

本项目是一个基于深度学习的视网膜血管分割系统，支持UNet和UNet++两种模型架构，每种架构都可以使用EfficientNet或ResNet作为backbone。系统主要用于医学图像处理中的视网膜血管分割任务，可以帮助医生更准确地诊断视网膜相关疾病。

## 项目结构

```
graduation thesis/
├── src/                      # 源代码文件夹
│   ├── __pycache__/         # 缓存文件
│   ├── results/             # 结果目录
│   │   ├── evaluation/      # 评估结果
│   │   ├── generalization_test/  # 泛化测试结果
│   │   ├── unet_efficientnet_train/  # UNet with EfficientNet训练结果
│   │   ├── unet_resnet_train/  # UNet with ResNet训练结果
│   │   ├── unetpp_efficientnet_train/  # UNet++ with EfficientNet训练结果
│   │   ├── unetpp_resnet_train/  # UNet++ with ResNet训练结果
│   │   └── comprehensive_evaluation_report.txt  # 综合评估报告
│   ├── callbacks.py         # 回调函数
│   ├── config.py            # 配置参数
│   ├── data_loader.py       # 数据加载和预处理
│   ├── evaluate_model_unet.py  # UNet评估脚本
│   ├── evaluate_model_unetpp.py  # UNet++评估脚本
│   ├── requirements.txt     # 依赖包列表
│   ├── test_generalization_unet.py  # UNet泛化测试脚本
│   ├── test_generalization_unetpp.py  # UNet++泛化测试脚本
│   ├── unet_model.py        # UNet模型定义
│   ├── unet_train.py        # UNet训练脚本
│   ├── unetpp_model.py      # UNet++模型定义
│   └── unetpp_train.py      # UNet++训练脚本
├── data/                   # 数据集文件夹
│   ├── CHASE_DB1/          # CHASE_DB1数据集
│   ├── DRIVE/              # DRIVE数据集
│   ├── HRF/                # HRF数据集
│   └── STARE/              # STARE数据集
├── 用户手册.md              # 用户手册
├── 维护手册.md              # 维护手册
└── README.md             # 项目说明文档
```

## 模型架构

本项目实现了两种模型架构，每种架构都支持两种backbone：

### 1. UNet模型
- **EfficientNet V2 B2 backbone**：轻量级模型，适合资源受限环境
- **ResNet50 backbone**：更深层的模型，可能获得更好的性能

### 2. UNet++模型
- **EfficientNet V2 B2 backbone**：轻量级模型，适合资源受限环境
- **ResNet50 backbone**：更深层的模型，可能获得更好的性能

## 安装依赖

首先安装必要的依赖包：

```bash
cd src
pip install -r requirements.txt
```

## 训练模型

### 训练UNet模型

- 使用EfficientNet backbone（默认）：
  ```bash
  cd src
  python unet_train.py
  ```

- 使用ResNet backbone：
  ```bash
  cd src
  python unet_train.py --backbone resnet
  ```

- 指定自定义结果目录：
  ```bash
  cd src
  python unet_train.py --backbone resnet --results-dir custom_results
  ```

### 训练UNet++模型

- 使用EfficientNet backbone（默认）：
  ```bash
  cd src
  python unetpp_train.py
  ```

- 使用ResNet backbone：
  ```bash
  cd src
  python unetpp_train.py --backbone resnet
  ```

- 指定自定义结果目录：
  ```bash
  cd src
  python unetpp_train.py --backbone resnet --results-dir custom_results
  ```

## 评估模型

### 评估UNet模型

- 评估EfficientNet backbone模型（默认）：
  ```bash
  cd src
  python evaluate_model_unet.py
  ```

- 评估ResNet backbone模型：
  ```bash
  cd src
  python evaluate_model_unet.py --backbone resnet
  ```

### 评估UNet++模型

- 评估EfficientNet backbone模型（默认）：
  ```bash
  cd src
  python evaluate_model_unetpp.py
  ```

- 评估ResNet backbone模型：
  ```bash
  cd src
  python evaluate_model_unetpp.py --backbone resnet
  ```

## 测试泛化能力

### 测试UNet模型泛化能力

- 测试EfficientNet backbone模型（默认）：
  ```bash
  cd src
  python test_generalization_unet.py
  ```

- 测试ResNet backbone模型：
  ```bash
  cd src
  python test_generalization_unet.py --backbone resnet
  ```

### 测试UNet++模型泛化能力

- 测试EfficientNet backbone模型（默认）：
  ```bash
  cd src
  python test_generalization_unetpp.py
  ```

- 测试ResNet backbone模型：
  ```bash
  cd src
  python test_generalization_unetpp.py --backbone resnet
  ```

## 训练配置

- **数据集**: DRIVE视网膜血管分割数据集
- **模型**: 
  - UNet with EfficientNet V2 B2 backbone
  - UNet with ResNet50 backbone
  - UNet++ with EfficientNet V2 B2 backbone
  - UNet++ with ResNet50 backbone
- **图像大小**: 512x512
- **批次大小**: 6
- **训练轮数**: 1001
- **学习率**: 0.0001
- **预测可视化**: 每40个epoch生成一次预测图片

## 输出结果

### 训练结果

训练结果保存在以下目录：
- UNet with EfficientNet：`src/results/unet_efficientnet_train/`
- UNet with ResNet：`src/results/unet_resnet_train/`
- UNet++ with EfficientNet：`src/results/unetpp_efficientnet_train/`
- UNet++ with ResNet：`src/results/unetpp_resnet_train/`

每个目录包含：
- `best_model.keras`：最佳模型权重
- `training_summary.txt`：训练总结
- `training_record.json`：详细训练记录
- `loss_curve_large.png`：损失曲线
- `iou_curve_large.png`：IoU曲线
- `dice_curve_large.png`：Dice曲线
- `prediction_epoch_*.png`：不同 epoch 的预测结果

### 评估结果

评估结果保存在以下目录：
- UNet with EfficientNet：`src/results/evaluation/unet/EfficientNet/`
- UNet with ResNet：`src/results/evaluation/unet/ResNet/`
- UNet++ with EfficientNet：`src/results/evaluation/unetpp/EfficientNet/`
- UNet++ with ResNet：`src/results/evaluation/unetpp/ResNet/`

每个目录包含：
- `evaluation_report.txt`：评估报告
- `evaluation_visualization_*.png`：评估可视化结果

### 泛化测试结果

泛化测试结果保存在以下目录：
- UNet with EfficientNet：`src/results/generalization_test/unet/EfficientNet/`
- UNet with ResNet：`src/results/generalization_test/unet/ResNet/`
- UNet++ with EfficientNet：`src/results/generalization_test/unetpp/EfficientNet/`
- UNet++ with ResNet：`src/results/generalization_test/unetpp/RESNET/`

每个目录包含：
- `generalization_summary.txt`：泛化测试总结
- `CHASE_DB1_evaluation_report.txt`：CHASE_DB1数据集评估报告
- `HRF_evaluation_report.txt`：HRF数据集评估报告
- `*_visualization_*.png`：可视化结果

### 综合评估报告

综合评估报告保存在：`src/results/comprehensive_evaluation_report.txt`

该报告包含所有模型在DRIVE验证集和泛化测试集上的性能对比。

## 配置参数

可以在`src/config.py`中修改训练参数：

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

## 数据集路径

确保数据集已放置在正确的位置：

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

## 注意事项

- 训练过程中会自动创建相应的结果目录
- 预测图片会每40个epoch保存一次
- 最佳模型会根据验证集dice分数自动保存
- 训练使用了早停机制，如果200个epoch没有改善会自动停止
- UNet++模型包含更多的网络层，参数量较大，训练时间更长
- 泛化测试会在CHASE_DB1和HRF数据集上评估模型性能

## 文档

- **用户手册**：`用户手册.md` - 详细的使用说明
- **维护手册**：`维护手册.md` - 系统实现细节和维护指南

