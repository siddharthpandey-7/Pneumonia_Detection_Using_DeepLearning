import os
import base64
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -------------------- ABSOLUTE PATH FIX (CRITICAL) --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# -------------------- MODEL CONFIG --------------------
MODEL_URL = "https://huggingface.co/siddharthpandey7/pneumonia-model/resolve/main/best_vgg19_pneumonia.h5"
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
IMG_SIZE = (128, 128)

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from HuggingFace...")
        r = requests.get(MODEL_URL)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Model downloaded successfully.")

download_model()

print("Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("Model loaded successfully.")

# -------------------- ROUTES --------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("result.html", prediction="No file uploaded", confidence=0)

    file = request.files["file"]
    if file.filename == "":
        return render_template("result.html", prediction="No file selected", confidence=0)

    # Save uploaded file
    upload_dir = os.path.join(BASE_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    file.save(file_path)

    # Convert image to base64 (for display)
    pil_image = Image.open(file_path).convert("RGB")
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    img_data = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()

    # Preprocess image
    img = load_img(file_path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Prediction
    normal_prob, pneumonia_prob = model.predict(arr)[0]

    if pneumonia_prob > normal_prob:
        label = "PNEUMONIA DETECTED"
        confidence = pneumonia_prob * 100
    else:
        label = "NORMAL"
        confidence = normal_prob * 100

    return render_template(
        "result.html",
        prediction=label,
        confidence=round(confidence, 2),
        img_data=img_data
    )

@app.route("/health")
def health():
    return "OK", 200

# -------------------- ENTRY POINT --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
