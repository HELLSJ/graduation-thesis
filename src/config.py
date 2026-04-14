# Configuration parameters

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
