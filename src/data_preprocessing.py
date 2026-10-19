import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

def load_and_preprocess(batch_size=32, img_size=(48, 48)):
    """
    Load train, validation, and test datasets from folders.
    Folder structure should be:
    Datasets/data/train/<class_name>/
    Datasets/data/val/<class_name>/
    Datasets/data/test/<class_name>/
    """
    
    train_dir = "../data/Datasets/data/train"
    val_dir = "../data/Datasets/data/val"
    test_dir = "../data/Datasets/data/test"
    
    train_ds = image_dataset_from_directory(
        train_dir,
        label_mode="categorical",
        batch_size=batch_size,
        image_size=img_size,
        color_mode="grayscale",
        shuffle=True
    )
    
    val_ds = image_dataset_from_directory(
        val_dir,
        label_mode="categorical",
        batch_size=batch_size,
        image_size=img_size,
        color_mode="grayscale",
        shuffle=False
    )
    
    test_ds = image_dataset_from_directory(
        test_dir,
        label_mode="categorical",
        batch_size=batch_size,
        image_size=img_size,
        color_mode="grayscale",
        shuffle=False
    )
    
    # Normalize pixel values to [0,1]
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
    
    return train_ds, val_ds, test_ds
