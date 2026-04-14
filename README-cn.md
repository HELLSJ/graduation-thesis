# 🩸 视网膜血管分割项目

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.12.0-red?logo=keras&logoColor=white)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/HELLSJ/graduation-thesis?style=social)](https://github.com/HELLSJ/graduation-thesis)

**基于深度学习的视网膜血管分割系统**

[English](README.md) | [中文文档](#概述)

</div>

---

## 📋 概述

本项目是一个基于深度学习的视网膜血管分割系统，支持 **UNet** 和 **UNet++** 两种模型架构，每种架构都可以使用 **EfficientNet** 或 **ResNet** 作为backbone。系统主要用于医学图像处理中的视网膜血管分割任务，可以帮助医生更准确地诊断视网膜相关疾病。

### ✨ 主要特性

- 🔬 **多种模型架构**：支持UNet和UNet++，配合不同的backbone
- 🎯 **高性能表现**：UNet++ with EfficientNet在DRIVE验证集上达到0.6539的Dice分数
- 📊 **全面评估**：包含训练、验证和多数据集泛化测试
- 🖼️ **可视化展示**：自动生成训练曲线和预测结果
- 📁 **有序输出**：结构化的结果目录和详细报告

---

## 🏗️ 模型架构

### 支持的模型

| 模型 | Backbone | 参数量 | 描述 |
|-------|----------|------------|-------------|
| **UNet** | EfficientNet V2 B2 | ~9.2M | 轻量级模型，平衡性好 |
| **UNet** | ResNet50 | ~32.5M | 更深的模型，特征丰富 |
| **UNet++** | EfficientNet V2 B2 | ~9.2M | 先进架构，嵌套跳跃连接 |
| **UNet++** | ResNet50 | ~32.5M | 最强大的组合，适合复杂分割 |

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    输入图像 (512x512x3)                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    编码器 (Backbone)                         │
│  ┌─────────────────┐        ┌─────────────────┐            │
│  │ EfficientNet    │   OR   │ ResNet50        │            │
│  │ V2 B2           │        │                 │            │
│  └─────────────────┘        └─────────────────┘            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    解码器与跳跃连接                           │
│         (UNet 或 UNet++ 架构)                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    输出掩码 (512x512x1)                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 项目结构

```
graduation thesis/
├── src/                              # 源代码
│   ├── results/                      # 训练和评估结果
│   │   ├── evaluation/               # 评估结果
│   │   ├── generalization_test/      # 泛化测试结果
│   │   ├── unet_efficientnet_train/  # UNet + EfficientNet结果
│   │   ├── unet_resnet_train/        # UNet + ResNet结果
│   │   ├── unetpp_efficientnet_train/ # UNet++ + EfficientNet结果
│   │   └── unetpp_resnet_train/      # UNet++ + ResNet结果
│   ├── callbacks.py                  # 训练回调函数
│   ├── config.py                     # 配置参数
│   ├── data_loader.py                # 数据加载工具
│   ├── evaluate_model_unet.py        # UNet评估脚本
│   ├── evaluate_model_unetpp.py      # UNet++评估脚本
│   ├── test_generalization_unet.py   # UNet泛化测试
│   ├── test_generalization_unetpp.py # UNet++泛化测试
│   ├── unet_model.py                 # UNet模型定义
│   ├── unet_train.py                 # UNet训练脚本
│   ├── unetpp_model.py               # UNet++模型定义
│   └── unetpp_train.py               # UNet++训练脚本
├── data/                             # 数据集目录
│   ├── CHASE_DB1/                    # CHASE_DB1数据集
│   ├── DRIVE/                        # DRIVE数据集
│   └── HRF/                          # HRF数据集
├── user-manual.md                    # 用户手册（英文）
├── user-manual-cn.md                 # 用户手册（中文）
├── maintenance-manual.md             # 维护手册（英文）
├── maintenance-manual-cn.md          # 维护手册（中文）
└── README.md                         # 项目说明文档
```

---

## 🚀 快速开始

### 环境要求

- Python 3.7或更高版本
- TensorFlow 2.20.0
- CUDA支持的GPU（推荐）

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/HELLSJ/graduation-thesis.git
   cd graduation-thesis
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **安装依赖**
   ```bash
   cd src
   pip install -r requirements.txt
   ```

### 训练模型

#### UNet模型

```bash
# 使用EfficientNet backbone（默认）
python unet_train.py

# 使用ResNet backbone
python unet_train.py --backbone resnet

# 自定义结果目录
python unet_train.py --backbone resnet --results-dir custom_results
```

#### UNet++模型

```bash
# 使用EfficientNet backbone（默认）
python unetpp_train.py

# 使用ResNet backbone
python unetpp_train.py --backbone resnet
```

### 评估模型

```bash
# 评估UNet模型
python evaluate_model_unet.py --backbone resnet

# 评估UNet++模型
python evaluate_model_unetpp.py --backbone resnet
```

### 泛化测试

```bash
# 测试UNet模型泛化能力
python test_generalization_unet.py --backbone resnet

# 测试UNet++模型泛化能力
python test_generalization_unetpp.py --backbone resnet
```

---

## 📊 训练配置

| 参数 | 值 | 描述 |
|-----------|-------|-------------|
| **数据集** | DRIVE | 视网膜血管分割数据集 |
| **图像大小** | 512×512 | 输入图像尺寸 |
| **批次大小** | 6 | 训练批次大小 |
| **训练轮数** | 1001 | 最大训练轮数 |
| **学习率** | 0.0001 | Adam优化器学习率 |
| **早停机制** | 200轮 | 无改善时停止 |
| **预测间隔** | 40轮 | 每N轮保存预测结果 |

---

## 📈 训练输出

每次训练会生成以下文件：

| 文件 | 描述 |
|------|-------------|
| `best_model.keras` | 基于验证集Dice分数的最佳模型权重 |
| `training_summary.txt` | 可读的训练总结 |
| `training_record.json` | JSON格式的详细训练指标 |
| `loss_curve_large.png` | 训练和验证损失曲线 |
| `iou_curve_large.png` | 训练和验证IoU曲线 |
| `dice_curve_large.png` | 训练和验证Dice曲线 |
| `prediction_epoch_*.png` | 不同轮次的预测结果 |

---

## ⚙️ 配置

编辑 `src/config.py` 自定义训练参数：

```python
# 图像和批次设置
IMAGE_SIZE = 512
BATCH_SIZE = 6

# 数据集路径
ROOT_DIR = 'data'
TRAIN_IMAGES_DIR = 'DRIVE/training/images'
TRAIN_MASKS_DIR = 'DRIVE/training/1st_manual'

# 模型参数
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
N_CLASSES = 1
ACTIVATION = 'sigmoid'

# 训练参数
EPOCHS = 1001
LEARNING_RATE = 0.0001

# Backbone选项
ENCODER_NAME = 'efficientnet_v2_b2'
ENCODER_NAME_REFIX = 'resnet50'
ENCODER_DEPTH = 4
```

---

## 📦 数据集结构

确保数据集按以下结构组织：

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

## 📝 注意事项

- ✅ 训练过程中会自动创建相应的结果目录
- ✅ 预测图片会每40个epoch保存一次
- ✅ 最佳模型会根据验证集Dice分数自动保存
- ✅ 早停机制防止过拟合
- ⚠️ UNet++参数量更大，训练时间更长
- ⚠️ 泛化测试会在未见过的数据集上评估模型性能

---

## 📚 文档

| 文档 | 语言 | 描述 |
|----------|----------|-------------|
| [用户手册](user-manual.md) | English | 详细使用说明 |
| [用户手册](user-manual-cn.md) | 中文 | 详细使用说明 |
| [维护手册](maintenance-manual.md) | English | 系统实现细节 |
| [维护手册](maintenance-manual-cn.md) | 中文 | 系统实现细节和维护指南 |

---

## 🤝 贡献

欢迎贡献！请随时提交Pull Request。

---

## 🙏 致谢

- [DRIVE数据集](https://drive.grand-challenge.org/) - 数字视网膜图像血管提取
- [CHASE_DB1数据集](https://blogs.kingston.ac.uk/retinal/chasedb1/) - 英国儿童心脏与健康研究
- [HRF数据集](https://www5.cs.fau.de/research/data/fundus-images/) - 高分辨率眼底图像数据库

---

<div align="center">

**用 ❤️ 为医学图像分析而制作**

[⬆ 返回顶部](#-视网膜血管分割项目)

</div>
