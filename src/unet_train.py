"""
UNet training script

Usage:
  # Train UNet with EfficientNet backbone (default):
  cd src && python unet_train.py
  
  # Train UNet with ResNet backbone:
  cd src && python unet_train.py --backbone resnet
  
  # Specify custom results directory:
  cd src && python unet_train.py --backbone resnet --results-dir custom_results
"""
import os
import sys
import time
import json
import matplotlib.pyplot as plt
import argparse

import warnings
import keras
from config import EPOCHS, LEARNING_RATE
from data_loader import get_train_valid_datasets
from unet_model import build_model, compile_model
from callbacks import DisplayCallback

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

# Set backend
os.environ["KERAS_BACKEND"] = "tensorflow"


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train UNet model with different backbones')
    parser.add_argument('--backbone', type=str, default='efficientnet',
                      choices=['efficientnet', 'resnet'],
                      help='Backbone model to use (default: efficientnet)')
    parser.add_argument('--results-dir', type=str, default=None,
                      help='Directory to save results (default: auto-generated)')
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    backbone = args.backbone
    
    # Generate results directory based on backbone if not specified
    if args.results_dir:
        RESULTS_DIR = args.results_dir
    else:
        if backbone == 'resnet':
            RESULTS_DIR = 'results/unet_resnet_train'
        else:
            RESULTS_DIR = 'results/unet_efficientnet_train'
    
    # Create results directory if it doesn't exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print(f"Training UNet with {backbone} backbone")
    print(f"Results will be saved to: {RESULTS_DIR}")
    
    print("Loading datasets...")
    train_dataset, valid_dataset = get_train_valid_datasets()

    print("Building model...")
    model = build_model(backbone=backbone)
    model_params = model.count_params() / 1e6

    print("Compiling model...")
    model = compile_model(model, LEARNING_RATE)

    print("Setting up callbacks...")
    callbacks = [
        DisplayCallback(valid_dataset, RESULTS_DIR, epoch_interval=40),
        keras.callbacks.ModelCheckpoint(
            os.path.join(RESULTS_DIR, 'best_model.keras'),
            monitor='val_dice',
            save_best_only=True,
            mode='max'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_dice',
            patience=200,
            mode='max'
        )
    ]

    print("Starting training...")
    print(f"Predictions will be saved every 40 epochs to {RESULTS_DIR}/")

    # Start timing
    start_time = time.time()

    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # End timing
    end_time = time.time()
    training_time = end_time - start_time

    print("Training completed!")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Best model saved to {os.path.join(RESULTS_DIR, 'best_model.keras')}")

    # Plot training curves
    plot_training_curves(history, RESULTS_DIR, model, training_time, backbone)

    # Save training record for later use
    save_training_record(history, model, training_time, RESULTS_DIR, backbone)


def save_training_record(history, model, training_time, save_dir, backbone):
    """Save training record for later testing and Grad-CAM visualization"""
    print("Saving training record...")

    # Determine model name based on backbone
    if backbone == 'resnet':
        model_name = 'UNet with ResNet Encoder'
    else:
        model_name = 'UNet with EfficientNet Encoder'

    # Extract final metrics
    record = {
        "model_info": {
            "name": model_name,
            "backbone": backbone,
            "parameters": model.count_params(),
            "parameters_m": model.count_params() / 1e6,
            "input_shape": list(model.input_shape[1:]) if model.input_shape else None,
            "output_shape": list(model.output_shape[1:]) if model.output_shape else None
        },
        "training_info": {
            "total_epochs": len(history.history['loss']),
            "training_time_seconds": training_time,
            "training_time_minutes": training_time / 60,
            "training_time_hours": training_time / 3600
        },
        "final_metrics": {
            "train": {
                "loss": float(history.history['loss'][-1]),
                "iou": float(history.history['iou'][-1]),
                "dice": float(history.history['dice'][-1])
            },
            "validation": {
                "loss": float(history.history['val_loss'][-1]),
                "iou": float(history.history['val_iou'][-1]),
                "dice": float(history.history['val_dice'][-1])
            }
        },
        "best_metrics": {
            "best_val_dice": float(max(history.history['val_dice'])),
            "best_val_dice_epoch": int(history.history['val_dice'].index(max(history.history['val_dice']))) + 1,
            "best_val_iou": float(max(history.history['val_iou'])),
            "best_val_iou_epoch": int(history.history['val_iou'].index(max(history.history['val_iou']))) + 1
        },
        "full_history": {
            "loss": [float(x) for x in history.history['loss']],
            "iou": [float(x) for x in history.history['iou']],
            "dice": [float(x) for x in history.history['dice']],
            "val_loss": [float(x) for x in history.history['val_loss']],
            "val_iou": [float(x) for x in history.history['val_iou']],
            "val_dice": [float(x) for x in history.history['val_dice']]
        },
        "model_paths": {
            "best_model": os.path.join(save_dir, 'best_model.keras'),
            "training_curves": os.path.join(save_dir, 'training_curves.png'),
            "loss_curve": os.path.join(save_dir, 'loss_curve_large.png'),
            "iou_curve": os.path.join(save_dir, 'iou_curve_large.png'),
            "dice_curve": os.path.join(save_dir, 'dice_curve_large.png')
        }
    }

    # Save to JSON file
    record_path = os.path.join(save_dir, 'training_record.json')
    with open(record_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=4, ensure_ascii=False)

    print(f"Training record saved to {record_path}")

    # Also save a summary text file
    summary_path = os.path.join(save_dir, 'training_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"Training Summary - {model_name}\n")
        f.write("=" * 60 + "\n\n")

        f.write("[Model Information]\n")
        f.write(f"  Model Name: {record['model_info']['name']}\n")
        f.write(f"  Backbone: {record['model_info']['backbone']}\n")
        f.write(f"  Parameters: {record['model_info']['parameters_m']:.3f}M\n")
        f.write(f"  Input Shape: {record['model_info']['input_shape']}\n")
        f.write(f"  Output Shape: {record['model_info']['output_shape']}\n\n")

        f.write("[Training Information]\n")
        f.write(f"  Total Epochs: {record['training_info']['total_epochs']}\n")
        f.write(f"  Training Time: {record['training_info']['training_time_minutes']:.2f} minutes\n\n")

        f.write("[Final Metrics]\n")
        f.write("  Training:\n")
        f.write(f"    Loss: {record['final_metrics']['train']['loss']:.4f}\n")
        f.write(f"    IoU:  {record['final_metrics']['train']['iou']:.4f}\n")
        f.write(f"    Dice: {record['final_metrics']['train']['dice']:.4f}\n")
        f.write("  Validation:\n")
        f.write(f"    Loss: {record['final_metrics']['validation']['loss']:.4f}\n")
        f.write(f"    IoU:  {record['final_metrics']['validation']['iou']:.4f}\n")
        f.write(f"    Dice: {record['final_metrics']['validation']['dice']:.4f}\n\n")

        f.write("[Best Metrics]\n")
        f.write(f"  Best Val Dice: {record['best_metrics']['best_val_dice']:.4f} (Epoch {record['best_metrics']['best_val_dice_epoch']})\n")
        f.write(f"  Best Val IoU:  {record['best_metrics']['best_val_iou']:.4f} (Epoch {record['best_metrics']['best_val_iou_epoch']})\n\n")

        f.write("[Model Paths]\n")
        f.write(f"  Best Model: {record['model_paths']['best_model']}\n")
        f.write(f"  Training Curves: {record['model_paths']['training_curves']}\n\n")

        f.write("=" * 60 + "\n")

    print(f"Training summary saved to {summary_path}")

    return record


def plot_training_curves(history, save_dir, model, training_time, backbone):
    """Plot training curves for metrics with improved styling"""
    print("Plotting training curves...")

    # Extract metrics
    dice = history.history['dice']
    iou = history.history['iou']
    loss = history.history['loss']
    val_dice = history.history['val_dice']
    val_iou = history.history['val_iou']
    val_loss = history.history['val_loss']

    model_params = model.count_params() / 1e6
    
    # Determine model name based on backbone
    if backbone == 'resnet':
        model_name = 'UNet with ResNet Encoder'
    else:
        model_name = 'UNet with EfficientNet Encoder'

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create comprehensive summary figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # Loss plot (large, top left)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(loss, label='Train Loss', color='#1f77b4', linewidth=2)
    ax1.plot(val_loss, label='Validation Loss', color='#ff7f0e', linewidth=2)
    ax1.set_title('Loss over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # IoU plot (large, top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(iou, label='Train IoU', color='#2ca02c', linewidth=2)
    ax2.plot(val_iou, label='Validation IoU', color='#d62728', linewidth=2)
    ax2.set_title('IoU over Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('IoU', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Dice plot (large, middle left)
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(dice, label='Train Dice', color='#9467bd', linewidth=2)
    ax3.plot(val_dice, label='Validation Dice', color='#8c564b', linewidth=2)
    ax3.set_title('Dice over Epochs', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Dice', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Training metrics summary box
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    train_text = f"""
    Training Metrics Summary

    Final Train Loss: {loss[-1]:.4f}
    Final Train IoU: {iou[-1]:.4f}
    Final Train Dice: {dice[-1]:.4f}
    """
    ax4.text(0.1, 0.5, train_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # Validation metrics summary box
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis('off')
    val_text = f"""
    Validation Metrics Summary

    Final Val Loss: {val_loss[-1]:.4f}
    Final Val IoU: {val_iou[-1]:.4f}
    Final Val Dice: {val_dice[-1]:.4f}
    """
    ax5.text(0.1, 0.5, val_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    # Model information box
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    model_text = f"""
    Model Information

    Model: {model_name}
    Parameters: {model_params:.3f}M
    Training Time: {training_time:.2f}s
    Total Epochs: {len(dice)}
    """
    ax6.text(0.1, 0.5, model_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

    # Performance gap analysis
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    gap_text = f"""
    Performance Gap Analysis

    Loss Gap: {loss[-1] - val_loss[-1]:.4f}
    IoU Gap: {iou[-1] - val_iou[-1]:.4f}
    Dice Gap: {dice[-1] - val_dice[-1]:.4f}
    """
    ax7.text(0.1, 0.5, gap_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

    plt.suptitle(f'{model_name} Training Summary', fontsize=16, fontweight='bold', y=0.98)

    # Save comprehensive plot
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Training curves saved to {plot_path}")
    plt.close()

    # Generate three separate large curve plots

    # 1. Loss Curve (Large)
    plt.figure(figsize=(14, 8))
    plt.plot(loss, label='Train Loss', color='#1f77b4', linewidth=2.5)
    plt.plot(val_loss, label='Validation Loss', color='#ff7f0e', linewidth=2.5)
    plt.title(f'Loss Curve - {model_name}', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(save_dir, 'loss_curve_large.png')
    plt.savefig(loss_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Loss curve saved to {loss_path}")
    plt.close()

    # 2. IoU Curve (Large)
    plt.figure(figsize=(14, 8))
    plt.plot(iou, label='Train IoU', color='#2ca02c', linewidth=2.5)
    plt.plot(val_iou, label='Validation IoU', color='#d62728', linewidth=2.5)
    plt.title(f'IoU Curve - {model_name}', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('IoU', fontsize=14)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    iou_path = os.path.join(save_dir, 'iou_curve_large.png')
    plt.savefig(iou_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"IoU curve saved to {iou_path}")
    plt.close()

    # 3. Dice Curve (Large)
    plt.figure(figsize=(14, 8))
    plt.plot(dice, label='Train Dice', color='#9467bd', linewidth=2.5)
    plt.plot(val_dice, label='Validation Dice', color='#8c564b', linewidth=2.5)
    plt.title(f'Dice Curve - {model_name}', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Dice', fontsize=14)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    dice_path = os.path.join(save_dir, 'dice_curve_large.png')
    plt.savefig(dice_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Dice curve saved to {dice_path}")
    plt.close()

    # Print final metrics
    print("\n" + "=" * 50)
    print("Final Metrics Summary")
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"Model Parameters: {model_params:.3f}M")
    print(f"Training Time: {training_time:.2f} seconds ({training_time / 60:.2f} minutes)")
    print(f"Total Epochs: {len(dice)}")
    print("-" * 50)
    print("Training Metrics:")
    print(f"  Loss: {loss[-1]:.4f}")
    print(f"  IoU:  {iou[-1]:.4f}")
    print(f"  Dice: {dice[-1]:.4f}")
    print("-" * 50)
    print("Validation Metrics:")
    print(f"  Loss: {val_loss[-1]:.4f}")
    print(f"  IoU:  {val_iou[-1]:.4f}")
    print(f"  Dice: {val_dice[-1]:.4f}")
    print("-" * 50)
    print("Performance Gap:")
    print(f"  Loss: {loss[-1] - val_loss[-1]:.4f}")
    print(f"  IoU:  {iou[-1] - val_iou[-1]:.4f}")
    print(f"  Dice: {dice[-1] - val_dice[-1]:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
