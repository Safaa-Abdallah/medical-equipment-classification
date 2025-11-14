# 🏥 Medical Equipment Classification using CNN

A deep learning model that classifies surgical equipment in operating rooms using Convolutional Neural Networks (CNN). This project assists medical robots in identifying surgical tools with 72% accuracy.

## 📊 Project Overview
- **Model:** Custom CNN
- **Accuracy:** 72%
- **Classes:** 3 (Operation Table, Surgery Light, Operation Room)
- **Dataset:** 251 images
- **Framework:** TensorFlow/Keras

📈 Results

The model achieved 72% accuracy in classifying three types of surgical equipment, demonstrating potential for real-world medical applications

## 🛠️ Installation
```bash
git clone https://github.com/Safaa-Abdallah/medical-equipment-classification.git
cd medical-equipment-classification
pip install -r requirements.txt
🚀 Usage
import tensorflow as tf
model = tf.keras.models.load_model('model.h5')
prediction = model.predict(image_array)
📁 Project Structure
medical-equipment-classification/
├── notebooks/          # Jupyter/Colab notebooks
├── samples/           # Sample images
├── requirements.txt   # Dependencies
└── README.md         # Project documentation
👩‍💻 Author

Safaa Kamaleldin Izzeldin Abdallah

· GitHub: Safaa-Abdallah