"""
UNet++ evaluation script

Usage:
  # Evaluate UNet++ with EfficientNet backbone (default):
  cd src && python evaluate_model_unetpp.py
  
  # Evaluate UNet++ with ResNet backbone:
  cd src && python evaluate_model_unetpp.py --backbone resnet
  
  # Specify custom results directory:
  cd src && python evaluate_model_unetpp.py --backbone resnet --results-dir custom_results
  
  # Specify custom model directory:
  cd src && python evaluate_model_unetpp.py --backbone resnet --model-dir path/to/model
"""
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import medicai
from medicai.utils import GradCAM
from unetpp_model import build_improved_unet, compile_improved_model
from config import LEARNING_RATE, ROOT_DIR, TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set backend
os.environ["KERAS_BACKEND"] = "tensorflow"


def _decode_tiff_pil(file_path):
    """使用PIL读取TIFF文件"""
    path_str = file_path.numpy().decode('utf-8')
    from PIL import Image
    img = Image.open(path_str)
    return np.array(img, dtype=np.float32)


def read_files(file_path, mask=False):
    """读取文件并进行预处理"""
    if mask:
        image = tf.io.read_file(file_path)
        image = tf.io.decode_gif(image)  # out: (1, h, w, 3)
        image = tf.squeeze(image)  # out: (h, w, 3)
        image = tf.image.rgb_to_grayscale(image)  # out: (h, w, 1)
        image = tf.divide(image, 128)
        image.set_shape([None, None, 1])
        image = tf.image.resize(
            images=image,
            size=[512, 512],
            method='nearest'
        )
        image = tf.cast(
            tf.cast(image, tf.int32), tf.float32
        )
    else:
        # 使用tf.py_function包装PIL的调用
        [image] = tf.py_function(
            _decode_tiff_pil, [file_path], [tf.float32]
        )
        image = image[:,:,:3]  # out: (h, w, 3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(
            images=image, size=[512, 512]
        )
        image = image / 255.
    return image


def load_data(image_list, mask_list):
    """加载数据"""
    image = read_files(image_list)
    mask  = read_files(mask_list, mask=True)
    return image, mask


def data_generator(image_list, mask_list, augment=False):
    """创建数据生成器"""
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(6, drop_remainder=augment)

    if augment:
        # 定义 augmentation 层
        aug_layers = [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        ]

        def augment_data(x, y):
            z = tf.concat([x, y], axis=-1)
            for layer in aug_layers:
                z = layer(z)

            x = z[..., :3]
            y = z[..., 3:]
            y = tf.cast(
                tf.cast(y, tf.int32), tf.float32
            )
            return x, y

        dataset = dataset.map(
            augment_data, num_parallel_calls=tf.data.AUTOTUNE
        )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def get_test_dataset():
    """获取测试数据集"""
    # 加载训练数据
    images = sorted([
        os.path.join(ROOT_DIR, TRAIN_IMAGES_DIR, fname)
        for fname in os.listdir(os.path.join(ROOT_DIR, TRAIN_IMAGES_DIR))
        if fname.endswith(('jpg', 'png', 'tif'))
    ])
    
    masks = sorted([
        os.path.join(ROOT_DIR, TRAIN_MASKS_DIR, fname)
        for fname in os.listdir(os.path.join(ROOT_DIR, TRAIN_MASKS_DIR))
        if fname.endswith(('jpg', 'png', 'tif', 'gif'))
    ])
    
    # 创建验证数据集（使用训练集的最后10张图像）
    test_images = images[-10:]
    test_masks = masks[-10:]
    test_dataset = data_generator(test_images, test_masks)
    print(f"Created test dataset with {len(test_images)} images")
    return test_dataset


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate UNet++ model with different backbones')
    parser.add_argument('--backbone', type=str, default='efficientnet',
                      choices=['efficientnet', 'resnet'],
                      help='Backbone model to use (default: efficientnet)')
    parser.add_argument('--results-dir', type=str, default=None,
                      help='Directory to save results (default: auto-generated)')
    parser.add_argument('--model-dir', type=str, default=None,
                      help='Directory where the trained model is saved (default: auto-generated)')
    return parser.parse_args()


def load_trained_model(backbone, model_dir):
    """Load the trained UNet++ model"""
    if not model_dir:
        if backbone == 'resnet':
            model_dir = 'results/unetpp_resnet_train'
        else:
            model_dir = 'results/unetpp_efficientnet_train'
    
    model_path = os.path.join(model_dir, 'best_model.keras')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first.")
        sys.exit(1)
    
    # Build and compile the model architecture with additional metrics
    model = build_improved_unet(backbone=backbone)
    
    # Define optimizer with gradient clipping
    optim = keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)

    # Define hybrid loss
    loss_fn = medicai.losses.BinaryDiceCELoss(
        from_logits=False,
        num_classes=1
    )

    # Define metrics with recall and precision
    dice_metric = medicai.metrics.BinaryDiceMetric(
        name='dice',
        from_logits=False,
        ignore_empty=True,
        num_classes=1,
        threshold=0.5
    )
    iou_metric = keras.metrics.BinaryIoU(name='iou')
    recall_metric = keras.metrics.Recall(name='recall')
    precision_metric = keras.metrics.Precision(name='precision')
    
    metrics = [iou_metric, dice_metric, recall_metric, precision_metric]
    
    # Compile the model
    model.compile(
        optimizer=optim,
        loss=loss_fn,
        metrics=metrics
    )
    
    # Load the weights
    model.load_weights(model_path)
    print(f"Model loaded successfully from {model_path}")
    return model

def visualize_results(model, dataset, save_dir, backbone):
    """可视化评估结果"""
    # 初始化Grad-CAM
    cam = None
    try:
        # 使用与notebook相同的target_layer
        target_layer = 'decoder_stage1_conv_2_activation'
        
        # 检查layer是否存在
        layer_exists = False
        for layer in model.layers:
            if layer.name == target_layer:
                layer_exists = True
                break
        
        if not layer_exists:
            # 自动找到合适的卷积层作为备选
            print(f"Layer {target_layer} not found, searching for alternative...")
            target_layer = None
            for layer in reversed(model.layers):
                if len(layer.output_shape) == 4 and 'conv' in layer.name.lower():
                    target_layer = layer.name
                    print(f"Using alternative layer: {target_layer}")
                    break
            
            if target_layer is None:
                target_layer = model.layers[-2].name
                print(f"Using fallback layer: {target_layer}")
        
        # 创建Grad-CAM
        cam = GradCAM(
            model=model,
            target_layer=target_layer,
            task_type='auto'
        )
        print("Grad-CAM initialized successfully")
    except Exception as e:
        print(f"Error initializing Grad-CAM: {e}")
        cam = None
    
    # 获取测试数据
    x, y = next(iter(dataset))
    y_pred = model.predict(x)
    
    # 计算Grad-CAM
    heatmap = None
    if cam is not None:
        try:
            heatmap = cam.compute_heatmap(x)
        except Exception as e:
            print(f"Error computing Grad-CAM: {e}")
    
    # 可视化
    n = min(4, len(x))
    fig, axes = plt.subplots(n, 5, figsize=(16, 4 * n))
    
    for i in range(n):
        # 原始图像
        ax1 = axes[i, 0]
        ax1.imshow(x[i])
        ax1.set_title(f"Image {i+1}")
        ax1.axis("off")
        
        # 真实掩码
        ax2 = axes[i, 1]
        ax2.imshow(y[i], cmap='gray')
        ax2.set_title(f"GT Mask {i+1}")
        ax2.axis("off")
        
        # 预测掩码
        ax3 = axes[i, 2]
        ax3.imshow((y_pred[i] > 0.5).astype("int32"), cmap='gray')
        ax3.set_title(f"Pred Mask {i+1}")
        ax3.axis("off")
        
        # Grad-CAM
        ax4 = axes[i, 3]
        if heatmap is not None:
            ax4.imshow(heatmap[i], cmap='jet')
        else:
            ax4.imshow(np.zeros_like(x[i][:, :, 0]), cmap='jet')
        ax4.set_title(f"GradCAM {i+1}")
        ax4.axis("off")
        
        # 预测叠加在原始图像上
        ax5 = axes[i, 4]
        ax5.imshow(x[i])
        ax5.imshow((y_pred[i] > 0.5).astype("int32"), cmap='hot', alpha=0.4)
        ax5.set_title(f"Overlay {i+1}")
        ax5.axis("off")
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'evaluation_visualization_unetpp_{backbone}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {save_path}")


def evaluate_model():
    """Evaluate the UNet++ model on the test dataset"""
    # Parse command line arguments
    args = parse_args()
    backbone = args.backbone
    
    # Generate results directory based on backbone if not specified
    if args.results_dir:
        RESULTS_DIR = args.results_dir
    else:
        if backbone == 'resnet':
            RESULTS_DIR = 'results/evaluation/unetpp/ResNet'
        else:
            RESULTS_DIR = 'results/evaluation/unetpp/EfficientNet'
    
    # Create results directory if it doesn't exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print(f"Evaluating UNet++ with {backbone} backbone")
    print(f"Results will be saved to: {RESULTS_DIR}")
    
    print("Loading test dataset...")
    test_dataset = get_test_dataset()
    
    print("Loading trained model...")
    model = load_trained_model(backbone, args.model_dir)
    
    print("Evaluating model...")
    results = model.evaluate(test_dataset)
    
    loss = results[0]
    iou = results[1]
    dice = results[2]
    recall = results[3]
    precision = results[4]
    
    print(f"Test Loss: {loss:.4f}")
    print(f"Test IoU: {iou:.4f}")
    print(f"Test Dice: {dice:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test Precision: {precision:.4f}")
    
    # Save evaluation results
    save_evaluation_results(loss, iou, dice, recall, precision, RESULTS_DIR, backbone)
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(model, test_dataset, RESULTS_DIR, backbone)
    
    return loss, iou, dice, recall, precision


def save_evaluation_results(loss, iou, dice, recall, precision, save_dir, backbone):
    """Save evaluation results to a file"""
    # Determine model name based on backbone
    model_name = f'UNet++ with {backbone.capitalize()} Backbone'
    
    report_path = os.path.join(save_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Model Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("[Model Information]\n")
        f.write(f"  Model: {model_name}\n")
        f.write(f"  Backbone: {backbone}\n")
        f.write("  Optimizer: Adam with gradient clipping\n")
        f.write("  Loss: BinaryDiceCELoss\n\n")
        
        f.write("[Evaluation Results]\n")
        f.write(f"  Loss: {loss:.4f}\n")
        f.write(f"  IoU:  {iou:.4f}\n")
        f.write(f"  Dice: {dice:.4f}\n")
        f.write(f"  Recall: {recall:.4f}\n")
        f.write(f"  Precision: {precision:.4f}\n\n")
        
        f.write("[Test Dataset]\n")
        f.write("  Number of images: 10\n")
        f.write("  Image size: 512x512\n")
        f.write("=" * 60 + "\n")
    
    print(f"Evaluation report saved to {report_path}")


if __name__ == "__main__":
    evaluate_model()
