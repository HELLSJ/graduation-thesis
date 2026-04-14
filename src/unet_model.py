"""
UNet model definition

Usage:
  # This file is imported by unet_train.py
  # For training UNet with EfficientNet backbone:
  # python src/unet_train.py
  #
  # For training UNet with ResNet backbone:
  # python src/unet_train.py --backbone resnet
"""
import keras
import medicai
from config import INPUT_SHAPE, N_CLASSES, ACTIVATION, ENCODER_DEPTH

def build_model(backbone='efficientnet'):
    """Build UNet model with specified backbone
    
    Args:
        backbone: Backbone name ('efficientnet' or 'resnet')
        
    Returns:
        keras.Model: Built UNet model
    """
    # Clear session to free up RAM
    keras.backend.clear_session()
    
    # Choose encoder based on backbone
    if backbone.lower() == 'resnet':
        encoder_name = 'resnet50'
        model_name = 'UNet with ResNet Encoder'
    else:  # default to efficientnet
        encoder_name = 'efficientnet_v2_b2'
        model_name = 'UNet with EfficientNet Encoder'
    
    # Build the model using MedicAI
    model = medicai.models.UNet(
        encoder_name=encoder_name,
        encoder_depth=ENCODER_DEPTH,
        input_shape=INPUT_SHAPE,
        num_classes=N_CLASSES,
        classifier_activation=ACTIVATION,
    )
    
    model_params = model.count_params() / 1e6
    print(f"Model: {model_name}")
    print(f"Model parameters: {model_params:.3f}M")
    
    return model

def compile_model(model, learning_rate):
    """Compile UNet model with standard settings
    
    Args:
        model: keras.Model to compile
        learning_rate: Learning rate for optimizer
        
    Returns:
        keras.Model: Compiled model
    """
    # Define optimizer
    optim = keras.optimizers.Adam(learning_rate=learning_rate)

    # Define hybrid loss
    loss_fn = medicai.losses.BinaryDiceCELoss(
        from_logits=False,
        num_classes=N_CLASSES
    )

    # Define metrics
    metrics = [
        keras.metrics.BinaryIoU(name='iou'),
        medicai.metrics.BinaryDiceMetric(
            name='dice',
            from_logits=False,
            ignore_empty=True,
            num_classes=N_CLASSES
        )
    ]

    # Compile the model
    model.compile(
        optimizer=optim,
        loss=loss_fn,
        metrics=metrics
    )

    return model
