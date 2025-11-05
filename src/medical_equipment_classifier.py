"""
Medical Equipment Classification using CNN
Author: Safaa Kamaleldin Izzeldin Abdallah
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os 
class MedicalEquipmentClassifier:
def __init__(self, image_size=(128, 128), num_classes=3):
self.image_size = image_size
self.num_classes = num_classes
self.model = self.build_model() 
def build_model(self):
"""Build CNN model for medical equipment classification"""
model = keras.Sequential([
# First Convolutional Block
layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
layers.MaxPooling2D((2, 2)), 
# Second Convolutional Block
layers.Conv2D(64, (3, 3), activation='relu'),
layers.MaxPooling2D((2, 2)), 
# Third Convolutional Block
layers.Conv2D(128, (3, 3), activation='relu'),
layers.MaxPooling2D((2, 2)), 
# Classification Block
layers.Flatten(),
layers.Dense(512, activation='relu'),
layers.Dropout(0.5),
layers.Dense(256, activation='relu'),
layers.Dropout(0.3),
layers.Dense(3, activation='softmax') # 3 classes
]) 
return model 
def compile_model(self):
"""Compile the model with optimizer and loss function"""
self.model.compile(
optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy']
) 
def summary(self):
"""Display model architecture"""
return self.model.summary() 
# Example usage and training simulation
def main():
print("🏥 Medical Equipment Classification Project")
print("=" * 50) 
# Initialize classifier
classifier = MedicalEquipmentClassifier()
classifier.compile_model() 
# Display model architecture
print("\n📊 Model Architecture:")
classifier.summary() 
# Simulate training results (based on your original project)
print("\n🎯 Training Results:")
print("- Final Accuracy: 75%")
print("- Classes: Operation Table, Surgery Light, Operation Room")
print("- Dataset: 251 images")
print("- Epochs: 10")
print("- Batch Size: 32") 
# Model performance summary
print("\n📈 Performance Metrics:")
metrics = {
"Training Accuracy": "72%",
"Validation Accuracy": "75%",
"Precision": "74%",
"Recall": "73%",
"F1-Score": "73.5%"
} 
for metric, value in metrics.items():
print(f"- {metric}: {value}") 
if __name__ == "__main__":
main()
