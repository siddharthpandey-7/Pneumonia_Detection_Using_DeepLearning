# 🩺 Pneumonia Detection Using Deep Learning

An end-to-end Deep Learning–based web application for detecting pneumonia from chest X-ray images using CNN transfer learning with **VGG19**.  
The trained model is deployed as a **Flask web application** using **Docker** on **Hugging Face Spaces (Free CPU tier)**, enabling real-time predictions with confidence scores.

---

## 🚀 Live Demo

🔗 **Deployed Application (Free):**  
https://siddharthpandey7-pneumonia-detection-flask.hf.space


---

## 📌 Project Overview

This project demonstrates a complete **machine learning lifecycle**:

**Dataset → Preprocessing → CNN Training → Evaluation → Model Saving → Deployment → Inference**

A convolutional neural network based on **VGG19 (ImageNet pretrained)** is trained on chest X-ray images to classify scans into:

- **NORMAL**
- **PNEUMONIA**

The trained model is integrated into a Flask-based web application where users can upload an X-ray image and instantly view predictions along with confidence scores.

---

## 🎯 Key Features

- Pneumonia detection from chest X-ray images using CNNs
- Transfer learning with **VGG19**
- Two-phase training: frozen base model + fine-tuning
- Final test accuracy **~91.7%**
- High pneumonia recall (**~95%**), critical for medical screening tasks
- Real-time inference via **Flask**
- Model served from **Hugging Face Hub** (no local model storage)
- Deployed using **Docker** on **Hugging Face Spaces (Free CPU tier)**
- Simple, clean, and user-friendly web interface

---

## 📊 Dataset

- **Source**: [Kaggle – Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- **Classes**: NORMAL, PNEUMONIA
- **Image Type**: Grayscale chest X-rays (converted to RGB)

### Dataset Split

| Split      | Normal | Pneumonia | Total |
|------------|--------|-----------|-------|
| Train      | 1,341  | 3,875     | 5,216 |
| Validation | 8      | 8         | 16    |
| Test       | 234    | 390       | 624   |

> **Note**: Due to the small validation set, final performance is primarily evaluated on the test set.

---

## 🔄 Data Preprocessing

- Resize images to **128 × 128**
- Convert grayscale images to 3-channel RGB
- Normalize pixel values to [0, 1]
- Apply data augmentation (rotation, shift, zoom, horizontal flip)

These steps help improve generalization and reduce overfitting.

---

## 🧠 Model Architecture

- **Base Model**: VGG19 (ImageNet pretrained, top layers removed)
- **Custom Classification Head**:
  - Flatten
  - Dense (512, ReLU) + Dropout
  - Dense (128, ReLU)
  - Dense (2, Softmax)

### Training Strategy

**Phase 1 – Transfer Learning**
- VGG19 layers frozen
- Classifier layers trained
- Adam optimizer (lr = 1e-4)

**Phase 2 – Fine-Tuning**
- Upper VGG19 layers unfrozen
- Lower learning rate (1e-5)
- Early stopping and learning-rate scheduling applied

---

## 📈 Model Performance (Test Set)

- **Final Test Accuracy**: 91.67%
- **Final Test Loss**: 0.2557

### Key Observations

- **Pneumonia Recall**: ~95%
- Reduced false negatives after fine-tuning
- Balanced trade-off between sensitivity and specificity

---

## 🌐 Web Application & Deployment

### Inference Flow

1. User uploads a chest X-ray image
2. Image is resized and normalized
3. Pretrained CNN performs inference (no retraining)
4. Prediction label and confidence score are displayed

### Deployment Details

- **Backend**: Flask, TensorFlow / Keras
- **Frontend**: HTML, CSS, JavaScript
- **Containerization**: Docker
- **Hosting**: Hugging Face Spaces (CPU Basic – Free tier)
- **Model Hosting**: Hugging Face Hub (downloaded at runtime)

---

## 📁 Project Structure
```
pneumonia-detection/
│
├── main.py                    # Flask application (inference)
├── Dockerfile                 # Docker configuration
├── requirements.txt
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   └── style.css
│
├── training_notebook.ipynb    # Model training & evaluation (Kaggle)
└── README.md
```

---

## 🛠️ Tools & Technologies

- Python
- TensorFlow / Keras
- Convolutional Neural Networks (CNN)
- Transfer Learning (VGG19)
- Flask
- Docker
- Hugging Face Spaces
- NumPy, Pandas
- Matplotlib, Seaborn
- HTML, CSS, JavaScript

---

## 📬 Contact & Support

- **GitHub**: https://github.com/siddharthpandey-7/Pneumonia_Detection
- **Email**: siddharthpandey97825@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/siddharth-kumar-pandey-003065343/

Feel free to ⭐ this repository if you find it helpful!

---

**Built with ❤️ for learning and applied AI in healthcare**
