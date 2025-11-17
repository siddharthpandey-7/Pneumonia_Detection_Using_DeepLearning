from flask import Flask, render_template, request, send_from_directory
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import requests, os, io
from PIL import Image

app = Flask(__name__)

# -------------------- MODEL DOWNLOAD --------------------
MODEL_URL = "https://huggingface.co/siddharthpandey7/pneumonia-model/resolve/main/best_vgg19_pneumonia.keras"
MODEL_PATH = "model.keras"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from HuggingFace...")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Model downloaded successfully.")

download_model()

print("Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("Model loaded.")

IMG_SIZE = (128, 128)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("result.html", prediction="No file uploaded")

    file = request.files["file"]
    if file.filename == "":
        return render_template("result.html", prediction="No file selected")

    # Save to /tmp folder (Render allows only this)
    file_path = os.path.join("/tmp", file.filename)
    file.save(file_path)

    # Preprocess
    img = load_img(file_path, target_size=IMG_SIZE)
    img_arr = img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    prediction = model.predict(img_arr)[0]  # [normal_prob, pneumonia_prob]
    normal_prob, pneumonia_prob = prediction

    if pneumonia_prob > normal_prob:
        label = "PNEUMONIA"
        confidence = pneumonia_prob * 100
    else:
        label = "NORMAL"
        confidence = normal_prob * 100

    confidence = round(confidence, 2)

    return render_template(
        "result.html",
        prediction=label,
        confidence=confidence,
        filename=file.filename,
    )

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory("/tmp", filename)

@app.route("/health")
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run()
