"""
UNet++ model definition

Usage:
  # This file is imported by unetpp_train.py
  # For training UNet++ with EfficientNet backbone:
  # python src/unetpp_train.py
  #
  # For training UNet++ with ResNet backbone:
  # python src/unetpp_train.py --backbone resnet
"""
import keras
import medicai
from keras import layers
from config import INPUT_SHAPE, N_CLASSES, ACTIVATION, ENCODER_NAME, ENCODER_NAME_REFIX, ENCODER_DEPTH

def build_improved_unet(backbone='efficientnet'):
    """Build UNet++ model with specified backbone
    
    Args:
        backbone: Backbone name ('efficientnet' or 'resnet')
        
    Returns:
        keras.Model: Built UNet++ model
    """
    keras.backend.clear_session()
    
    # Choose encoder based on backbone
    if backbone.lower() == 'resnet':
        encoder_name = ENCODER_NAME_REFIX
        model_name = 'UNet++ with ResNet Backbone'
    else:  # default to efficientnet
        encoder_name = ENCODER_NAME
        model_name = 'UNet++ with EfficientNet Backbone'
    
    # Use medicai's UNet++ implementation with optimized parameters and regularization
    model = medicai.models.UNetPlusPlus(
        encoder_name=encoder_name,
        encoder_depth=ENCODER_DEPTH,
        input_shape=INPUT_SHAPE,
        num_classes=N_CLASSES,
        classifier_activation=ACTIVATION,
        decoder_filters=(128, 64, 32, 16),  # Reduce decoder filters
        decoder_normalization="batch",  # Batch normalization as regularization
    )
    
    model_params = model.count_params() / 1e6
    print(f"Model: {model_name}")
    print(f"Model parameters: {model_params:.3f}M")
    print("Using optimized UNet++ architecture with regularization")
    
    return model

def compile_improved_model(model, learning_rate):
    """Compile model with same settings as original model"""
    # Define optimizer with gradient clipping for stability
    optim = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

    # Define hybrid loss - same as original model
    loss_fn = medicai.losses.BinaryDiceCELoss(
        from_logits=False,
        num_classes=N_CLASSES
    )

    # Define metrics - same as original model
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
