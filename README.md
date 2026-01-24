# 🩺 Pneumonia Detection Using Deep Learning

An end-to-end Deep Learning–based web application for detecting pneumonia from chest X-ray images using CNN transfer learning with VGG19. The trained model is deployed using Flask to provide real-time image predictions with confidence scores.


## 📌 Project Overview

This project demonstrates the complete machine learning workflow:

**Dataset → Preprocessing → CNN Training → Evaluation → Model Saving → Deployment → Prediction**

A convolutional neural network based on VGG19 (ImageNet pretrained) is trained on chest X-ray images to classify scans into:

- **NORMAL**
- **PNEUMONIA**

The trained model is integrated into a Flask web application where users can upload an X-ray image and instantly view the prediction.

## 🎯 Key Features

- Pneumonia detection from chest X-ray images using CNNs
- Transfer learning with **VGG19**
- Two-phase training: frozen base model + fine-tuning
- Final test accuracy **~91.7%**
- High pneumonia recall (**~95%**), important for medical screening
- Real-time inference using **Flask**
- Clear separation between training and inference
- Simple and user-friendly web interface

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

**Note**: The validation set is small; therefore, final performance assessment is based mainly on the test set.

## 🔄 Data Preprocessing

- Resize images to **128 × 128**
- Convert grayscale images to 3-channel RGB
- Normalize pixel values to [0, 1]
- Apply data augmentation during training (rotation, shift, zoom, flip)

These steps help improve generalization and reduce overfitting.

## 🧠 Model Architecture

- **Base Model**: VGG19 (pretrained on ImageNet, top layers removed)
- **Custom Head**:
  - Flatten
  - Dense (512, ReLU) + Dropout
  - Dense (128, ReLU)
  - Dense (2, Softmax)

### Training Strategy

**Phase 1 – Transfer Learning**
- VGG19 layers frozen
- Only classifier layers trained
- Adam optimizer (lr = 1e-4)

**Phase 2 – Fine-Tuning**
- Upper VGG19 layers unfrozen
- Lower learning rate (1e-5)
- Early stopping and learning-rate scheduling applied

## 📈 Model Performance (Test Set)

- **Final Test Accuracy**: 91.67%
- **Final Test Loss**: 0.2557

### Classification Highlights

- **Pneumonia Recall**: ~95%
- Fewer missed pneumonia cases after fine-tuning
- Acceptable trade-off between false positives and false negatives

## 🌐 Web Application (Inference)

### Prediction Flow

1. User uploads a chest X-ray image
2. Image is resized and normalized
3. Trained CNN performs inference (no retraining)
4. Prediction and confidence score are displayed

### Tech Stack

- **Backend**: Flask, TensorFlow/Keras
- **Frontend**: HTML, CSS, JavaScript

## 📁 Project Structure

```
pneumonia-detection/
│
├── app.py                  # Flask app (inference)
├── pneumonia.ipynb         # Training & evaluation notebook
├── model.h5                # Trained CNN model
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   └── style.css
│
├── requirements.txt
└── README.md
```

## 🚀 How to Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Open in browser:

```
http://localhost:8080
```

## 🛠️ Tools & Technologies

- Python
- TensorFlow / Keras
- Convolutional Neural Networks (CNN)
- Transfer Learning (VGG19)
- Flask
- NumPy, Pandas
- Matplotlib, Seaborn
- HTML, CSS, JavaScript


## 📬 Support

For questions, feedback, or contributions:

- **GitHub**: ('https://github.com/siddharthpandey-7/Pneumonia_Detection_Using_DeepLearning.git')
- **Email**: your.email@example.com
- **LinkedIn**: [Your Name](https://linkedin.com/in/yourprofile)

Feel free to ⭐ this repository if you find it helpful!


## ⚠️ Disclaimer

This project is not a medical device and should not be used for real-world diagnosis. It is intended strictly for learning, experimentation, and ML practice.

**Built with ❤️ for better healthcare through AI**
