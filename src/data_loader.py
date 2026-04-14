import os
import numpy as np
from PIL import Image
import tensorflow as tf
from config import IMAGE_SIZE, BATCH_SIZE, ROOT_DIR, TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR

# Define image extensions
exts = ('jpg', 'JPG', 'png', 'PNG', 'tif', 'gif', 'ppm')

# Define augmentation layers
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

def _decode_tiff_pil(file_path):
    path_str = file_path.numpy().decode('utf-8')
    img = Image.open(path_str)
    return np.array(img, dtype=np.float32)

def read_files(image_path, mask=False):
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.io.decode_gif(image) # out: (1, h, w, 3)
        image = tf.squeeze(image) # out: (h, w, 3)
        image = tf.image.rgb_to_grayscale(image) # out: (h, w, 1)
        # Apply threshold to only keep blood vessels (white pixels)
        image = tf.cast(image > 128, tf.float32)
        image.set_shape([None, None, 1])
        image = tf.image.resize(
            images=image, 
            size=[IMAGE_SIZE, IMAGE_SIZE], 
            method='nearest'
        )
    else:
        # image = tfio.experimental.image.decode_tiff(image) # out: (h, w, 4)
        [image] = tf.py_function(
            _decode_tiff_pil, [image_path], [tf.float32]
        )
        image = image[:,:,:3] # out: (h, w, 3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(
            images=image, size=[IMAGE_SIZE, IMAGE_SIZE]
        )
        image = image / 255.
    return image

def load_data(image_list, mask_list):
    image = read_files(image_list)
    mask  = read_files(mask_list, mask=True)
    return image, mask

def data_generator(image_list, mask_list, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=augment)

    if augment:
        dataset = dataset.map(
            augment_data, num_parallel_calls=tf.data.AUTOTUNE
        )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def get_train_valid_datasets():
    # Get training images and masks paths
    input_data = os.path.join(ROOT_DIR, TRAIN_IMAGES_DIR)
    images = sorted(
        [
            os.path.join(input_data, fname)
            for fname in os.listdir(input_data)
            if fname.endswith(exts) and not fname.startswith(".")
        ]
    )

    target_data = os.path.join(ROOT_DIR, TRAIN_MASKS_DIR)
    masks = sorted(
        [
            os.path.join(target_data, fname)
            for fname in os.listdir(target_data)
            if fname.endswith(exts) and not fname.startswith(".")
        ]
    )

    # Split into train and validation sets
    train_images, val_images = images[:-10], images[-10:]
    train_masks, val_masks = masks[:-10], masks[-10:]

    # Create datasets
    train_dataset = data_generator(train_images, train_masks, augment=True)
    valid_dataset = data_generator(val_images, val_masks)

    return train_dataset, valid_dataset