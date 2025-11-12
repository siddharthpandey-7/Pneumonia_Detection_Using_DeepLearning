import os
import requests
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# ✅ Hugging Face model link
MODEL_URL = "https://huggingface.co/siddharthpandey7/pneumonia-model/resolve/main/best_vgg19_pneumonia.h5"
MODEL_PATH = "best_vgg19_pneumonia.h5"

# ------------------ DOWNLOAD MODEL --------------------
def download_model():
    """Download the model from Hugging Face if not already present."""
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100000:
        print("🧠 Downloading model from Hugging Face...")
        response = requests.get(MODEL_URL, stream=True)

        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("✅ Model downloaded successfully!")
        else:
            raise Exception("❌ ERROR: Failed to download model from Hugging Face.")

# ------------------ LOAD MODEL --------------------
download_model()

try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    raise e

# ------------------ ROUTES --------------------
@app.route("/")
def index():
    """Home page route."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload and model prediction."""
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    try:
        img = Image.open(file).convert("RGB").resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)[0][0]
        result = "PNEUMONIA DETECTED" if prediction > 0.5 else "NORMAL"

        return render_template("result.html", result=result)
    except Exception as e:
        print("❌ Prediction error:", e)
        return "Error during prediction", 500

if __name__ == "__main__":
    # ✅ Important for Render — use 0.0.0.0 and port from env
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
