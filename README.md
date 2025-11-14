# Medical Equipment Classification using CNN
## Project Overview
A deep learning model that classifies surgical equipment in operating rooms using Convolutional Neural Networks (CNN). This project assists medical robots in identifying surgical tools with 75% accuracy.
## Features
- Image classification of 3 surgical equipment types
- Real-time predictions using CNN
- Multiple image processing techniques
- Data augmentation for improved accuracy
## Results
- Accuracy: 72%
- Classes: 3
- Dataset Size: 251 images
- Model: Custom CNN
## Installation Commands
git clone https://github.com/Safaa-Abdallah/medical-equipment-classification.git
pip install -r requirements.txt
## Quick Start Code
import tensorflow as tf
model = tf.keras.models.load_model('model.h5') prediction = model.predict(image_array)
## Author
Safaa Kamaleldin Izzeldin Abdallah
