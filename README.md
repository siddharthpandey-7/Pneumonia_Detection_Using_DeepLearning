# 🩺 Pneumonia Detection Using Deep Learning

An end-to-end Deep Learning–based web application that detects pneumonia from chest X-ray images using CNN transfer learning (VGG19). The system is trained on a public medical dataset and deployed using Flask for real-time inference with confidence scores.

## 🔍 Project Overview

This project demonstrates the complete Machine Learning lifecycle:

**Dataset → Preprocessing → CNN Training → Evaluation → Deployment → Prediction**

A VGG19-based CNN model is trained on chest X-ray images to classify scans as:
* **NORMAL**
* **PNEUMONIA**

The trained model is then integrated into a Flask web application, allowing users to upload an X-ray image and receive a prediction with confidence.

## 🎯 Key Highlights

* Transfer learning using **VGG19** (ImageNet pretrained)
* Two-phase training: frozen base + fine-tuning
* Strong performance with **~91.7% test accuracy**
* High pneumonia recall (**~95%**), critical for medical screening
* Real-time inference via **Flask**
* Clean and simple web UI for image upload and results

## 📊 Dataset

* **Source**: [Kaggle – Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
* **Classes**: NORMAL, PNEUMONIA
* **Image Type**: Grayscale chest X-rays (converted to RGB)

### Data Split

| Split      | Normal | Pneumonia | Total |
|------------|--------|-----------|-------|
| Train      | 1,341  | 3,875     | 5,216 |
| Validation | 8      | 8         | 16    |
| Test       | 234    | 390       | 624   |

⚠️ **Note**: The validation set is small; therefore, final conclusions are based primarily on test set performance.

## 🧠 Model Architecture

* **Base Model**: VGG19 (without top layers)
* **Custom Classifier Head**:
  * Flatten
  * Dense (512, ReLU) + Dropout (0.5)
  * Dense (128, ReLU)
  * Dense (2, Softmax)

### Training Strategy

**Phase 1 – Transfer Learning**
* VGG19 frozen
* Optimizer: Adam (lr = 1e-4)
* Early stopping applied

**Phase 2 – Fine-tuning**
* Unfrozen deeper VGG layers
* Lower learning rate (1e-5)
* Improved generalization and reduced misclassification

## 📈 Model Performance (Test Set)

* **Final Test Accuracy**: 91.67%
* **Pneumonia Recall**: ~95%
* **Final Test Loss**: 0.2557

### Confusion Matrix (After Fine-Tuning)

| Actual \ Predicted | NORMAL | PNEUMONIA |
|-------------------|--------|-----------|
| **NORMAL**        | 197    | 37        |
| **PNEUMONIA**     | 15     | 375       |

The model prioritizes high recall for pneumonia, minimizing missed positive cases, which is important in medical screening systems.

## 🌐 Web Application

### User Flow

1. User uploads a chest X-ray image
2. Image is resized and normalized
3. Trained CNN performs inference
4. Prediction and confidence score are displayed

### Backend

* **Flask** (Python)
* **TensorFlow / Keras** for inference
* Model loaded from `.h5` file at runtime

## 📁 Project Structure

```
pneumonia-detection/
│
├── app.py                  # Flask application (inference)
├── pneumonia.ipynb         # Model training & evaluation
├── model.h5                # Trained CNN model
│
├── templates/
│   ├── index.html          # Upload page
│   └── result.html         # Prediction results
│
├── static/
│   └── style.css           # UI styling
│
├── requirements.txt        # Dependencies
└── README.md
```

## 🚀 How to Run Locally

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run the application

```bash
python app.py
```

### 3️⃣ Open in browser

```
http://localhost:8080
```

## 🛠️ Tools & Technologies

* Python
* TensorFlow / Keras
* CNN & Transfer Learning
* Flask
* NumPy, Pandas
* Matplotlib, Seaborn
* HTML, CSS

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is not a medical device and should not be used for real clinical diagnosis. Predictions are intended as decision-support, not replacement for medical professionals.
