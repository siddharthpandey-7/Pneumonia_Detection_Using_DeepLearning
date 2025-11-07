from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# --- Initialize Flask app ---
app = Flask(__name__)

# Folder for uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
MODEL_PATH = 'best_vgg19_pneumonia.h5'
model = load_model(MODEL_PATH)

# Class names same as training order
class_names = ['NORMAL', 'PNEUMONIA']

# --- Homepage ---
@app.route('/')
def home():
    return render_template('index.html')

# --- Prediction route ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Save uploaded image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess image to match model input
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = round(100 * np.max(prediction[0]), 2)

    return render_template('result.html', 
                           filename=file.filename, 
                           prediction=predicted_class, 
                           confidence=confidence)

# --- Show uploaded image ---
@app.route('/display/<filename>')
def display_image(filename):
    return f'<img src="/static/uploads/{filename}" width="300">'

if __name__ == "__main__":
    app.run(debug=True)
