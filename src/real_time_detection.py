# real_time_detection.py (MINIMAL)

import cv2
import tensorflow as tf
import numpy as np

# Path to the model (optional reference file)
MODEL_PATH = "../data/Datasets/data/models/best_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Emotion list
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

print("This file is no longer used. Use ui_app.py for main application.")
