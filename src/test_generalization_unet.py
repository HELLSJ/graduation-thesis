"""
UNet generalization test script

Usage:
  # Test UNet with EfficientNet backbone (default):
  cd src && python test_generalization_unet.py
  
  # Test UNet with ResNet backbone:
  cd src && python test_generalization_unet.py --backbone resnet
  
  # Specify custom results directory:
  cd src && python test_generalization_unet.py --backbone resnet --results-dir custom_results
  
  # Specify custom model directory:
  cd src && python test_generalization_unet.py --backbone resnet --model-dir path/to/model
"""
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import keras
import medicai
from medicai.utils import GradCAM
import cv2
from unet_model import build_model, compile_model
from config import LEARNING_RATE, ROOT_DIR

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


def get_chase_db1_dataset():
    """加载CHASE_DB1数据集"""
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
    
    # 创建数据集
    dataset = data_generator(images, masks)
    print(f"Created CHASE_DB1 dataset with {len(images)} images")
    return dataset


def get_hrf_dataset():
    """加载HRF数据集"""
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
    
    # 创建数据集
    dataset = data_generator(images, masks)
    print(f"Created HRF dataset with {len(images)} images")
    return dataset


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test UNet model generalization with different backbones')
    parser.add_argument('--backbone', type=str, default='efficientnet',
                      choices=['efficientnet', 'resnet'],
                      help='Backbone model to use (default: efficientnet)')
    parser.add_argument('--results-dir', type=str, default=None,
                      help='Directory to save results (default: auto-generated)')
    parser.add_argument('--model-dir', type=str, default=None,
                      help='Directory where the trained model is saved (default: auto-generated)')
    return parser.parse_args()


def load_trained_model(backbone, model_dir):
    """Load the trained model"""
    if not model_dir:
        if backbone == 'resnet':
            model_dir = 'results/unet_resnet_train'
        else:
            model_dir = 'results/unet_efficientnet_train'
    
    model_path = os.path.join(model_dir, 'best_model.keras')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first.")
        sys.exit(1)
    
    # Build the model architecture
    model = build_model(backbone=backbone)
    
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
    recall_metric = keras.metrics.Recall(name='recall')
    precision_metric = keras.metrics.Precision(name='precision')
    
    metrics = [dice_metric, recall_metric, precision_metric]
    
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


def find_interesting_regions(pred_mask, gt_mask):
    pred_np = pred_mask.numpy() if hasattr(pred_mask, 'numpy') else pred_mask
    gt_np = gt_mask.numpy() if hasattr(gt_mask, 'numpy') else gt_mask
    pred_binary = (pred_np.squeeze() > 0.5).astype(np.uint8)
    gt_binary = (gt_np.squeeze() > 0.5).astype(np.uint8)
    diff = np.abs(pred_binary.astype(int) - gt_binary.astype(int))
    kernel = np.ones((15, 15), np.uint8)
    diff_dilated = cv2.dilate(diff.astype(np.uint8) * 255, kernel, iterations=1)
    contours, _ = cv2.findContours(diff_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 30 and h > 30:
            regions.append((x, y, w, h))
    return regions[:3]


def find_vessel_bifurcations(mask, min_distance=20):
    mask_np = mask.numpy() if hasattr(mask, 'numpy') else mask
    mask_uint8 = (mask_np.squeeze() * 255).astype(np.uint8)
    try:
        skeleton = cv2.ximgproc.thinning(mask_uint8)
    except AttributeError:
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask_uint8, kernel, iterations=1)
        skeleton = eroded
    corners = cv2.goodFeaturesToTrack(skeleton, maxCorners=10, qualityLevel=0.01, minDistance=min_distance)
    bifurcations = []
    if corners is not None:
        for corner in corners:
            x, y = corner.ravel()
            bifurcations.append((int(x), int(y)))
    return bifurcations


def find_thin_vessel_regions(mask, threshold=0.3):
    mask_np = mask.numpy() if hasattr(mask, 'numpy') else mask
    mask_uint8 = (mask_np.squeeze() * 255).astype(np.uint8)
    dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    thin_regions = dist_transform < (dist_transform.max() * threshold)
    contours, _ = cv2.findContours(thin_regions.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 20 and h > 20:
            regions.append((x, y, w, h))
    return regions[:5]


def compute_gradcam(model, image):
    target_layer = None
    for layer in reversed(model.layers):
        try:
            output_shape = layer.output.shape if hasattr(layer.output, 'shape') else None
            if output_shape is not None and len(output_shape) == 4 and 'conv' in layer.name.lower():
                target_layer = layer.name
                break
        except:
            continue
    if target_layer is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                target_layer = layer.name
                break
    if target_layer is None:
        target_layer = model.layers[-2].name
    try:
        cam = GradCAM(model=model, target_layer=target_layer, task_type='auto')
        heatmap = cam.compute_heatmap(image)
        return heatmap[0]
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return np.zeros((512, 512))


def visualize_results(model, dataset, save_dir, dataset_name, backbone):
    """增强版可视化评估结果"""
    x, y = next(iter(dataset))
    y_pred = model.predict(x)
    
    n = min(4, len(x))
    for i in range(n):
        image_np = x[i].numpy() if hasattr(x[i], 'numpy') else x[i]
        gt_mask_np = y[i].numpy() if hasattr(y[i], 'numpy') else y[i]
        pred_mask_np = y_pred[i]
        
        heatmap = compute_gradcam(model, tf.expand_dims(x[i], 0))
        interesting_regions = find_interesting_regions(pred_mask_np, gt_mask_np)
        bifurcations = find_vessel_bifurcations(gt_mask_np)
        thin_regions = find_thin_vessel_regions(gt_mask_np)
        
        fig = plt.figure(figsize=(24, 10))
        gs = fig.add_gridspec(2, 5, hspace=0.25, wspace=0.2)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image_np)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(gt_mask_np.squeeze(), cmap='gray')
        ax2.set_title('Ground Truth', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(pred_mask_np.squeeze(), cmap='gray')
        ax3.set_title(f'UNet-{backbone.capitalize()}\nPrediction', fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(heatmap, cmap='jet')
        ax4.set_title('Grad-CAM\nAttention Heatmap', fontsize=14, fontweight='bold')
        ax4.axis('off')
        
        ax5 = fig.add_subplot(gs[0, 4])
        ax5.imshow(image_np)
        ax5.imshow(heatmap, cmap='jet', alpha=0.5)
        ax5.set_title('Grad-CAM Overlay\n(Red=High Attention)', fontsize=14, fontweight='bold')
        ax5.axis('off')
        
        ax6 = fig.add_subplot(gs[1, 0])
        ax6.imshow(image_np)
        ax6.imshow(pred_mask_np.squeeze(), cmap='hot', alpha=0.4)
        ax6.set_title('Prediction Overlay', fontsize=14, fontweight='bold')
        ax6.axis('off')
        for j, (rx, ry, rw, rh) in enumerate(interesting_regions[:3]):
            rect = Rectangle((rx, ry), rw, rh, linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--')
            ax6.add_patch(rect)
            ax6.text(rx, ry-5, f'Detail {j+1}', color='yellow', fontsize=10, fontweight='bold')
        
        ax7 = fig.add_subplot(gs[1, 1])
        ax7.imshow(gt_mask_np.squeeze(), cmap='gray')
        for j, (bx, by) in enumerate(bifurcations[:5]):
            circle = plt.Circle((bx, by), 15, fill=False, color='cyan', linewidth=2)
            ax7.add_patch(circle)
            ax7.text(bx+20, by, f'B{j+1}', color='cyan', fontsize=9, fontweight='bold')
        ax7.set_title('Vessel Bifurcations\n(Cyan Circles)', fontsize=14, fontweight='bold')
        ax7.axis('off')
        
        ax8 = fig.add_subplot(gs[1, 2])
        ax8.imshow(gt_mask_np.squeeze(), cmap='gray')
        for j, (rx, ry, rw, rh) in enumerate(thin_regions[:3]):
            rect = Rectangle((rx, ry), rw, rh, linewidth=2, edgecolor='lime', facecolor='none')
            ax8.add_patch(rect)
            ax8.text(rx, ry-5, f'Thin {j+1}', color='lime', fontsize=10, fontweight='bold')
        ax8.set_title('Thin Vessel Regions\n(Green Boxes)', fontsize=14, fontweight='bold')
        ax8.axis('off')
        
        if len(interesting_regions) > 0:
            ax9 = fig.add_subplot(gs[1, 3])
            rx, ry, rw, rh = interesting_regions[0]
            margin = 10
            x1, y1 = max(0, rx-margin), max(0, ry-margin)
            x2, y2 = min(512, rx+rw+margin), min(512, ry+rh+margin)
            ax9.imshow(image_np[y1:y2, x1:x2])
            ax9.set_title(f'Zoomed Detail 1\n(Original)', fontsize=12, fontweight='bold')
            ax9.axis('off')
            
            ax10 = fig.add_subplot(gs[1, 4])
            detail_pred = pred_mask_np.squeeze()[y1:y2, x1:x2]
            ax10.imshow(detail_pred, cmap='gray')
            ax10.set_title(f'Zoomed Prediction\nDetail Region', fontsize=12, fontweight='bold')
            ax10.axis('off')
        else:
            for col in [3, 4]:
                ax = fig.add_subplot(gs[1, col])
                ax.text(0.5, 0.5, 'No significant\ndifference regions', ha='center', va='center', fontsize=14)
                ax.axis('off')
        
        save_path = os.path.join(save_dir, f'{dataset_name}_visualization_{backbone}_sample{i+1}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Visualization saved to: {save_path}")


def test_on_dataset(dataset_name, dataset, save_dir, backbone):
    """Test model on a specific dataset"""
    print(f"\nTesting on {dataset_name} dataset...")
    
    try:
        results = model.evaluate(dataset)
        loss = results[0]
        dice = results[1]
        recall = results[2]
        precision = results[3]
        
        print(f"{dataset_name} Loss: {loss:.4f}")
        print(f"{dataset_name} Dice: {dice:.4f}")
        print(f"{dataset_name} Recall: {recall:.4f}")
        print(f"{dataset_name} Precision: {precision:.4f}")
        
        # 保存评估结果
        report_path = os.path.join(save_dir, f'{dataset_name}_evaluation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"Model Evaluation Results on {dataset_name} Dataset\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("[Evaluation Metrics]\n")
            f.write(f"Loss: {loss:.4f}\n")
            f.write(f"Dice: {dice:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n\n")
            
            f.write("[Model Information]\n")
            f.write(f"Backbone: {backbone}\n")
            f.write("Optimizer: Adam with gradient clipping\n")
            f.write("Loss: BinaryDiceCELoss\n\n")
            
            f.write("[Dataset Information]\n")
            f.write(f"Number of images: {len(dataset)}\n")
            f.write("Image size: 512x512\n")
            f.write("=" * 60 + "\n")
        
        print(f"Evaluation report saved to: {report_path}")
        
        # 可视化结果
        print(f"\nGenerating visualizations for {dataset_name}...")
        visualize_results(model, dataset, save_dir, dataset_name, backbone)
        
        return loss, dice, recall, precision
    except Exception as e:
        print(f"Error testing on {dataset_name} dataset: {e}")
        return None, None, None, None


def test_generalization():
    """Test model generalization on different datasets"""
    global model
    
    # Parse command line arguments
    args = parse_args()
    backbone = args.backbone
    
    # Generate results directory based on backbone if not specified
    if args.results_dir:
        RESULTS_DIR = args.results_dir
    else:
        if backbone == 'resnet':
            RESULTS_DIR = 'results/generalization_test/unet/ResNet'
        else:
            RESULTS_DIR = 'results/generalization_test/unet/EfficientNet'
    
    # Create results directory if it doesn't exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print(f"Testing UNet with {backbone} backbone generalization")
    print(f"Results will be saved to: {RESULTS_DIR}")
    
    model = load_trained_model(backbone, args.model_dir)
    
    print("\n" + "=" * 70)
    print("Model Generalization Test")
    print("=" * 70)
    
    # Test on CHASE_DB1 dataset
    chase_db1_dataset = get_chase_db1_dataset()
    chase_loss, chase_dice, chase_recall, chase_precision = test_on_dataset("CHASE_DB1", chase_db1_dataset, RESULTS_DIR, backbone)
    
    # Test on HRF dataset
    hrf_dataset = get_hrf_dataset()
    hrf_loss, hrf_dice, hrf_recall, hrf_precision = test_on_dataset("HRF", hrf_dataset, RESULTS_DIR, backbone)
    
    # Save generalization results
    save_generalization_results(chase_loss, chase_dice, chase_recall, chase_precision, hrf_loss, hrf_dice, hrf_recall, hrf_precision, RESULTS_DIR, backbone)


def save_generalization_results(chase_loss, chase_dice, chase_recall, chase_precision, hrf_loss, hrf_dice, hrf_recall, hrf_precision, save_dir, backbone):
    """Save generalization test results"""
    # Determine model name based on backbone
    if backbone == 'resnet':
        model_name = 'UNet with ResNet Encoder'
    else:
        model_name = 'UNet with EfficientNet Encoder'
    
    report_path = os.path.join(save_dir, 'generalization_summary.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Model Generalization Test Results\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("[Model Information]\n")
        f.write(f"  Model: {model_name}\n")
        f.write(f"  Backbone: {backbone}\n")
        f.write("  Optimizer: Adam with gradient clipping\n")
        f.write("  Loss: BinaryDiceCELoss\n\n")
        
        f.write("[CHASE_DB1 Dataset]\n")
        if chase_loss is not None:
            f.write(f"Loss:     {chase_loss:.4f}\n")
            f.write(f"Dice:     {chase_dice:.4f}\n")
            f.write(f"Recall:   {chase_recall:.4f}\n")
            f.write(f"Precision: {chase_precision:.4f}\n")
        else:
            f.write("Test failed\n")
        f.write("\n")
        
        f.write("[HRF Dataset]\n")
        if hrf_loss is not None:
            f.write(f"Loss:     {hrf_loss:.4f}\n")
            f.write(f"Dice:     {hrf_dice:.4f}\n")
            f.write(f"Recall:   {hrf_recall:.4f}\n")
            f.write(f"Precision: {hrf_precision:.4f}\n")
        else:
            f.write("Test failed\n")
        f.write("\n")
        
        f.write("[Summary]\n")
        f.write("This report shows the performance of the UNet model on different retinal vessel segmentation datasets.\n")
        f.write("The model was trained on the DRIVE dataset and evaluated on CHASE_DB1 and HRF datasets.\n")
        f.write("Higher Dice coefficient, Recall, and Precision indicate better generalization ability.\n")
        f.write("Recall measures the model's ability to detect all real vessels.\n")
        f.write("Precision measures the accuracy of vessel predictions.\n")
        f.write("=" * 60 + "\n")
    
    print(f"\nGeneralization test results saved to {report_path}")


if __name__ == "__main__":
    test_generalization()
