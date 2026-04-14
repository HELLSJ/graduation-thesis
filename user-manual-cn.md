# 视网膜血管分割模型用户手册

## 1. 系统概述

本系统是一个基于深度学习的视网膜血管分割模型，支持UNet和UNet++两种模型架构，每种架构都可以使用EfficientNet或ResNet作为backbone。系统主要用于医学图像处理中的视网膜血管分割任务，可以帮助医生更准确地诊断视网膜相关疾病。

### 1.1 主要功能

- 训练视网膜血管分割模型
- 评估模型性能
- 测试模型的泛化能力
- 生成可视化结果和评估报告

### 1.2 系统架构

系统包含以下主要组件：
- 数据加载模块：负责加载和预处理数据集
- 模型定义模块：定义UNet和UNet++模型架构
- 训练模块：训练模型并保存最佳模型
- 评估模块：评估模型性能并生成报告
- 泛化测试模块：测试模型在不同数据集上的性能

## 2. 环境要求

### 2.1 硬件要求

- CPU：Intel Core i5或更高
- 内存：8GB或更多
- 存储空间：至少10GB
- GPU（可选）：支持CUDA的NVIDIA GPU，用于加速训练

### 2.2 软件要求

- 操作系统：Windows 10/11
- Python：3.7或更高版本
- 依赖库：
  - tensorflow
  - keras
  - medicai
  - pillow
  - numpy
  - matplotlib
  - pandas

## 3. 安装指南

### 3.1 克隆项目

将项目克隆到本地目录：

```bash
git clone https://github.com/HELLSJ/graduation-thesis.git
cd graduation-thesis
```
![[git-clone.png]]
### 3.2 创建虚拟环境

```bash
# 使用conda创建虚拟环境
conda create -n retinal-segmentation python=3.9
conda activate retinal-segmentation

# 或使用venv创建虚拟环境
python -m venv venv
# Windows激活
venv\Scripts\activate
# Linux/Mac激活
# source venv/bin/activate
```

### 3.3 安装依赖

```bash
cd src
pip install -r requirements.txt
```
![[environment_create.png]]
### 3.4 准备数据集

系统使用以下数据集：
- DRIVE：用于训练和验证
- CHASE_DB1：用于泛化测试
- HRF：用于泛化测试

请确保这些数据集已经下载并放置在`data`目录下，目录结构如下：

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

## 4. 使用指南

### 4.1 训练模型

#### 4.1.1 训练UNet模型

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

#### 4.1.2 训练UNet++模型

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

### 4.2 评估模型

#### 4.2.1 评估UNet模型

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

- 指定自定义结果目录：

```bash
cd src
python evaluate_model_unet.py --backbone resnet --results-dir custom_results
```

- 指定自定义模型目录：

```bash
cd src
python evaluate_model_unet.py --backbone resnet --model-dir path/to/model
```

#### 4.2.2 评估UNet++模型

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

### 4.3 测试泛化能力

#### 4.3.1 测试UNet模型泛化能力

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

#### 4.3.2 测试UNet++模型泛化能力

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

## 5. 结果查看

### 5.1 训练结果

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

### 5.2 评估结果

评估结果保存在以下目录：
- UNet with EfficientNet：`src/results/evaluation/unet/EfficientNet/`
- UNet with ResNet：`src/results/evaluation/unet/ResNet/`
- UNet++ with EfficientNet：`src/results/evaluation/unetpp/EfficientNet/`
- UNet++ with ResNet：`src/results/evaluation/unetpp/ResNet/`

每个目录包含：
- `evaluation_report.txt`：评估报告
- `evaluation_visualization_*.png`：评估可视化结果

### 5.3 泛化测试结果

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

### 5.4 综合评估报告

综合评估报告保存在：`src/results/comprehensive_evaluation_report.txt`

该报告包含所有模型在DRIVE验证集和泛化测试集上的性能对比。

## 6. 常见问题与解决方案

### 6.1 模型文件未找到

**问题**：运行评估脚本时提示"Model file not found"

**解决方案**：确保已经训练了对应的模型，训练命令如下：
- 训练UNet with ResNet：`python unet_train.py --backbone resnet`
- 训练UNet++ with ResNet：`python unetpp_train.py --backbone resnet`

### 6.2 内存不足

**问题**：训练时出现内存不足错误

**解决方案**：
- 减小批次大小（修改config.py中的BATCH_SIZE）
- 使用GPU进行训练
- 减小图像尺寸（修改config.py中的IMAGE_SIZE）

### 6.3 导入错误

**问题**：运行脚本时出现导入错误

**解决方案**：确保所有依赖项已正确安装，运行：
```bash
pip install -r requirements.txt
```

## 7. 示例运行

### 7.1 训练UNet with ResNet模型

```bash
cd src
python unet_train.py --backbone resnet
```
![[unet_resnet_train.png]]
### 7.2 评估UNet with ResNet模型

```bash
cd src
python evaluate_model_unet.py --backbone resnet
```
![[unet_resnet_eval.png]]
### 7.3 测试UNet with ResNet模型泛化能力

```bash
cd src
python test_generalization_unet.py --backbone resnet
```
![[unet_resnet_test.png]]