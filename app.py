import os
import requests
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# ✅ Hugging Face Model Details
MODEL_URL = "https://huggingface.co/siddharthpandey7/pneumonia-model/resolve/main/best_vgg19_pneumonia.h5"
MODEL_PATH = "best_vgg19_pneumonia.h5"

# ------------------ MODEL DOWNLOAD ------------------
def download_model():
    """Download model from Hugging Face if not already present."""
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
            raise Exception("❌ Failed to download model from Hugging Face.")

# ------------------ LAZY MODEL LOADING ------------------
model = None  # Model will load only when first needed

def get_model():
    """Load model only once and reuse (saves memory on Render)."""
    global model
    if model is None:
        print("🧩 Loading model into memory...")
        download_model()
        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded and ready!")
    return model

# ------------------ ROUTES ------------------
@app.route("/")
def home():
    """Render homepage."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload and perform prediction."""
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    try:
        # ✅ FIXED: match training input shape (128x128x3)
        img = Image.open(file).convert("RGB").resize((128, 128))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # ✅ Lazy load model
        model_instance = get_model()

        print("🧠 Predicting...")
        prediction = model_instance.predict(img)

        # ✅ Handle model output (binary or categorical)
        if prediction.ndim == 2 and prediction.shape[1] == 1:
            prob = float(prediction[0][0])
            result = "PNEUMONIA DETECTED" if prob > 0.5 else "NORMAL"
        elif prediction.ndim == 2 and prediction.shape[1] == 2:
            predicted_class = np.argmax(prediction)
            result = "PNEUMONIA DETECTED" if predicted_class == 1 else "NORMAL"
        else:
            result = "Prediction Error — Unexpected model output."

        print("✅ Prediction complete:", result)
        return render_template("result.html", result=result)

    except Exception as e:
        print("❌ Prediction error:", e)
        import traceback
        traceback.print_exc()
        return f"Error during prediction: {str(e)}", 500

# ------------------ MAIN ENTRY ------------------
if __name__ == "__main__":
    # ✅ Use 0.0.0.0 for Render + increase timeout via Procfile
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
