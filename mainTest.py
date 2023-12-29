import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

# Load the model
model = load_model('BrainTumor10EpochsCategorical.h5', custom_objects={'accuracy': 'accuracy'})

# Read and preprocess the image
image = cv2.imread('E:/BrainTumor Classification DL/datasets/yes/y3.jpg')
img = Image.fromarray(image)
img = img.resize((64, 64))
img = np.array(img)
input_img = np.expand_dims(img, axis=0)

# Predict the class probabilities
probs = model.predict(input_img)

# Find the index of the class with the highest probability
predicted_class = np.argmax(probs)

print(f"Predicted class: {predicted_class}")
print(f"Class probabilities: {probs}")
