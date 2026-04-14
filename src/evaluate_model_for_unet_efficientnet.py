import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


import medicai
from medicai.utils import GradCAM
from medicai.models import UNet
from medicai.losses import BinaryDiceCELoss
from medicai.metrics import BinaryDiceMetric

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

def load_training_data():
    """加载训练数据"""
    ROOT_DIR = 'd:\\graduation thesis\\data'
    TRAIN_IMAGES_DIR = 'DRIVE/training/images'
    TRAIN_MASKS_DIR = 'DRIVE/training/1st_manual'

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
    
    return images, masks

def evaluate_model(model_name='unet', model_path=None):
    """评估模型
    
    Args:
        model_name: 模型名称，支持 'unet' 和 'unetpp'
        model_path: 模型权重文件路径
    """
    # 配置
    if model_path is None:
        if model_name.lower() == 'unet':
            model_path = 'results/original/best_model.keras'
        elif model_name.lower() == 'unetpp':
            model_path = 'results/improved/best_model.keras'
        else:
            raise ValueError("Unsupported model name. Use 'unet' or 'unetpp'.")
    
    save_dir = f'results/evaluation/{model_name.lower()}'
    
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 加载训练数据
    train_images, train_masks = load_training_data()
    print(f"Loaded {len(train_images)} training images")
    
    # 创建验证数据集（使用训练集的最后10张图像）
    valid_images = train_images[-10:]
    valid_masks = train_masks[-10:]
    valid_dataset = data_generator(valid_images, valid_masks)
    print(f"Created validation dataset with {len(valid_images)} images")
    
    # 创建新模型并加载权重
    try:
        # 创建与训练时相同的模型
        input_shape=(512, 512, 3)
        n_classes=1
        activation='sigmoid'
        
        if model_name.lower() == 'unet':
            model = medicai.models.UNet(
                encoder_name='efficientnet_v2_b2',
                encoder_depth=4,
                input_shape=input_shape,
                num_classes=n_classes,
                classifier_activation=activation,
            )
        elif model_name.lower() == 'unetpp':
            model = medicai.models.UNetPlusPlus(
                encoder_name='efficientnet_v2_b2',
                encoder_depth=4,
                input_shape=input_shape,
                num_classes=n_classes,
                classifier_activation=activation,
                decoder_filters=(128, 64, 32, 16),
                decoder_normalization="batch",
            )
        
        # 编译模型
        optim = tf.keras.optimizers.Adam(0.0001)
        loss_fn = medicai.losses.BinaryDiceCELoss(
            from_logits=False,
            num_classes=n_classes
        )
        # 明确命名metrics
        # 使用 Dice、Recall 和 Precision 指标
        dice_metric = medicai.metrics.BinaryDiceMetric(
            name='dice',
            from_logits=False,
            ignore_empty=True,
            num_classes=n_classes,
            threshold=0.5
        )
        recall_metric = tf.keras.metrics.Recall(name='recall')
        precision_metric = tf.keras.metrics.Precision(name='precision')
        metrics = [dice_metric, recall_metric, precision_metric]
        model.compile(
            optimizer=optim,
            loss=loss_fn,
            metrics=metrics
        )
        
        # 尝试加载整个模型
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from: {model_path}")
        except Exception as e:
            print(f"Error loading entire model: {e}")
            print("Trying to load weights only...")
            # 尝试只加载权重
            model.load_weights(model_path)
            print("Weights loaded successfully")
    except Exception as e:
        print(f"Error creating/loading model: {e}")
        return
    
    # 评估模型
    print("\nEvaluating model...")
    results = model.evaluate(valid_dataset)
    print(f"Evaluation results: {results}")
    
    # 保存评估结果
    report_path = os.path.join(save_dir, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Model Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("[Evaluation Metrics]\n")
        if len(results) >= 4:
            f.write(f"Loss: {results[0]:.4f}\n")
            f.write(f"Dice: {results[1]:.4f}\n")
            f.write(f"Recall: {results[2]:.4f}\n")
            f.write(f"Precision: {results[3]:.4f}\n")
        elif len(results) >= 2:
            f.write(f"Loss: {results[0]:.4f}\n")
            f.write(f"Dice: {results[1]:.4f}\n")
        else:
            f.write(f"Dice: {results[0]:.4f}\n")
        f.write("\n")
        
        f.write("[Validation Dataset]\n")
        f.write(f"Number of images: {len(valid_images)}\n")
        f.write(f"Image size: 512x512\n")
        f.write("=" * 60 + "\n")
    
    # 可视化结果
    print("\nGenerating visualizations...")
    visualize_results(model, valid_dataset, save_dir)
    
    print(f"\nEvaluation completed! Results saved to: {save_dir}")

def visualize_results(model, dataset, save_dir):
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
    
    # 获取验证数据
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
    save_path = os.path.join(save_dir, 'evaluation_visualization.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {save_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate UNet or UNet++ model")
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'unetpp'],
                        help='Model to evaluate (default: unet)')
    parser.add_argument('--path', type=str, default=None,
                        help='Path to model weights file (default: auto-determined based on model name)')
    
    args = parser.parse_args()
    evaluate_model(model_name=args.model, model_path=args.path)
