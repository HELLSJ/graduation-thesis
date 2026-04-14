import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


import medicai
from medicai.utils import GradCAM
from medicai.models import UNet, UNetPlusPlus
from medicai.losses import BinaryDiceCELoss
from medicai.metrics import BinaryDiceMetric
from medicai.layers.drop_path import DropPath

def _decode_tiff_pil(file_path):
    """使用PIL读取TIFF文件"""
    path_str = file_path.numpy().decode('utf-8')
    from PIL import Image
    img = Image.open(path_str)
    return np.array(img, dtype=np.float32)

def read_files(file_path, mask=False):
    """读取文件并进行预处理"""
    if mask:
        # 直接使用PIL读取掩码，避免格式问题
        [image] = tf.py_function(
            _decode_tiff_pil, [file_path], [tf.float32]
        )

        # 处理不同维度的情况
        def process_image(img):
            # 检查维度
            if len(img.shape) == 2:
                # 2D图像，添加通道维度
                return tf.expand_dims(img, axis=-1)
            elif len(img.shape) == 3:
                # 3D图像，确保是单通道
                if img.shape[2] > 1:
                    return tf.image.rgb_to_grayscale(img)
                return img
            else:
                # 其他情况，返回原始图像
                return img
        
        # 使用tf.py_function处理形状
        [image] = tf.py_function(
            process_image, [image], [tf.float32]
        )
        
        # 设置形状
        image.set_shape([None, None, 1])
        
        # 归一化
        image = tf.divide(image, tf.reduce_max(image))
        
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
        
        # 处理不同维度的情况
        def process_image(img):
            if len(img.shape) == 2:
                # 2D图像，添加通道维度
                return tf.stack([img, img, img], axis=-1)
            elif len(img.shape) == 3:
                # 3D图像，确保是3通道
                if img.shape[2] == 1:
                    return tf.stack([img[:,:,0], img[:,:,0], img[:,:,0]], axis=-1)
                elif img.shape[2] > 3:
                    return img[:,:,:3]
                return img
            else:
                # 其他情况，返回原始图像
                return img
        
        # 使用tf.py_function处理形状
        [image] = tf.py_function(
            process_image, [image], [tf.float32]
        )
        
        # 设置形状
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
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def load_chase_db1_data():
    """加载CHASE_DB1数据集"""
    ROOT_DIR = 'd:\\graduation thesis\\data'
    IMAGES_DIR = 'CHASE_DB1/Images'
    MASKS_DIR = 'CHASE_DB1/Masks'
    
    # 加载图像
    images = sorted([
        os.path.join(ROOT_DIR, IMAGES_DIR, fname)
        for fname in os.listdir(os.path.join(ROOT_DIR, IMAGES_DIR))
        if fname.endswith(('jpg', 'png', 'tif'))
    ])
    
    # 加载掩码（使用1stHO标注）
    masks = []
    for img_path in images:
        img_name = os.path.basename(img_path)
        mask_name = img_name.replace('.jpg', '_1stHO.png')
        mask_path = os.path.join(ROOT_DIR, MASKS_DIR, mask_name)
        if os.path.exists(mask_path):
            masks.append(mask_path)
        else:
            print(f"Mask not found for {img_name}")
    
    return images, masks

def load_hrf_data():
    """加载HRF数据集"""
    ROOT_DIR = 'd:\\graduation thesis\\data'
    IMAGES_DIR = 'HRF/images'
    MASKS_DIR = 'HRF/manual1'
    
    # 加载图像
    images = sorted([
        os.path.join(ROOT_DIR, IMAGES_DIR, fname)
        for fname in os.listdir(os.path.join(ROOT_DIR, IMAGES_DIR))
        if fname.endswith(('jpg', 'png', 'tif', 'JPG'))
    ])
    
    # 加载掩码
    masks = []
    for img_path in images:
        img_name = os.path.basename(img_path)
        # 处理不同的文件扩展名
        if img_name.endswith('.JPG'):
            mask_name = img_name.replace('.JPG', '.tif')
        elif img_name.endswith('.jpg'):
            mask_name = img_name.replace('.jpg', '.tif')
        elif img_name.endswith('.png'):
            mask_name = img_name.replace('.png', '.tif')
        elif img_name.endswith('.tif'):
            mask_name = img_name
        else:
            mask_name = img_name + '.tif'
        
        mask_path = os.path.join(ROOT_DIR, MASKS_DIR, mask_name)
        if os.path.exists(mask_path):
            masks.append(mask_path)
        else:
            print(f"Mask not found for {img_name}")
            # 尝试其他可能的掩码名称
            alt_mask_name = img_name.split('.')[0] + '.tif'
            alt_mask_path = os.path.join(ROOT_DIR, MASKS_DIR, alt_mask_name)
            if os.path.exists(alt_mask_path):
                print(f"Found alternative mask: {alt_mask_name}")
                masks.append(alt_mask_path)
    
    return images, masks

def evaluate_model_on_dataset(dataset_name, model, save_dir, use_medicai):
    """在指定数据集上评估模型"""
    # 加载数据集
    if dataset_name == 'CHASE_DB1':
        images, masks = load_chase_db1_data()
    elif dataset_name == 'HRF':
        images, masks = load_hrf_data()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    print(f"Loaded {len(images)} images from {dataset_name} dataset")
    
    # 创建数据集
    dataset = data_generator(images, masks)
    
    # 评估模型
    print(f"\nEvaluating model on {dataset_name} dataset...")
    results = model.evaluate(dataset)
    print(f"{dataset_name} evaluation results: {results}")
    
    # 保存评估结果
    report_path = os.path.join(save_dir, f'{dataset_name}_evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"Model Evaluation Results on {dataset_name} Dataset\n")
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
        
        f.write("[Dataset Information]\n")
        f.write(f"Number of images: {len(images)}\n")
        f.write(f"Image size: 512x512\n")
        f.write("=" * 60 + "\n")
    
    # 可视化结果
    print(f"\nGenerating visualizations for {dataset_name}...")
    visualize_results(model, dataset, save_dir, dataset_name, use_medicai)
    
    return results

def visualize_results(model, dataset, save_dir, dataset_name, use_medicai):
    """可视化评估结果"""
    # 初始化Grad-CAM
    cam = None
    if use_medicai:
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
    
    # 获取数据
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
    save_path = os.path.join(save_dir, f'{dataset_name}_visualization.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {save_path}")

def main():
    """主函数"""
    # 配置
    model_path = 'results/original/best_model.keras'
    save_dir = 'results/generalization_test/UNET'
    model_type = 'unet'  # 'unet' or 'unetpp'
    
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 设置use_medicai标志
    global use_medicai
    use_medicai = True
    
    # 创建模型并加载权重
    try:
        # 创建与训练时相同的模型
        input_shape=(512, 512, 3)
        n_classes=1
        activation='sigmoid'
        
        # 只创建 UNet 模型
        model = medicai.models.UNet(
            encoder_name='efficientnet_v2_b2',
            encoder_depth=4,
            input_shape=input_shape,
            num_classes=n_classes,
            classifier_activation=activation,
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
    
    # 在不同数据集上评估模型
    datasets = ['CHASE_DB1', 'HRF']
    all_results = {}
    
    for dataset in datasets:
        results = evaluate_model_on_dataset(dataset, model, save_dir, use_medicai)
        all_results[dataset] = results
    
    # 生成综合报告
    report_path = os.path.join(save_dir, 'generalization_summary.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Model Generalization Test Results\n")
        f.write("=" * 60 + "\n\n")
        
        for dataset, results in all_results.items():
            f.write(f"[{dataset} Dataset]\n")
            if len(results) >= 4:
                loss = results[0]
                dice = results[1]
                recall = results[2]
                precision = results[3]
                f.write(f"Loss: {loss:.4f}\n")
                f.write(f"Dice: {dice:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"Precision: {precision:.4f}\n")
            elif len(results) >= 2:
                loss = results[0]
                dice = results[1]
                f.write(f"Loss: {loss:.4f}\n")
                f.write(f"Dice: {dice:.4f}\n")
            f.write("\n")
        
        f.write("[Summary]\n")
        f.write("This report shows the performance of the UNet model on different retinal vessel segmentation datasets.\n")
        f.write("The model was trained on the DRIVE dataset and evaluated on CHASE_DB1 and HRF datasets.\n")
        f.write("Higher Dice coefficient, Recall, and Precision indicate better generalization ability.\n")
        f.write("Recall measures the model's ability to detect all real vessels.\n")
        f.write("Precision measures the accuracy of vessel predictions.\n")
        f.write("=" * 60 + "\n")
    
    print(f"\nGeneralization test completed! Results saved to: {save_dir}")

if __name__ == "__main__":
    main()
