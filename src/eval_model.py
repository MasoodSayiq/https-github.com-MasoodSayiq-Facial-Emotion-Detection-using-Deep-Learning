# eval_model.py
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers import Rescaling
import os

# Paths
BASE_DIR = os.path.join("..", "data", "Datasets", "data")
TEST_DIR = os.path.join(BASE_DIR, "test")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.h5")

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Load test dataset
test_ds = image_dataset_from_directory(
    TEST_DIR,
    image_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    label_mode='categorical'
)

# Normalize pixel values
test_ds = test_ds.map(lambda x, y: (Rescaling(1./255)(x), y))
print("Test dataset loaded and normalized!")

# Evaluate the model
loss, accuracy = model.evaluate(test_ds)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

