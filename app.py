import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "ml_models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODELS_DIR, "model.keras")
TOKENIZER_FILE = os.path.join(MODELS_DIR, "tokenizer.pkl")
LE_FILE = os.path.join(MODELS_DIR, "label_encoder.pkl")

MAX_LEN = 300

app = Flask(__name__)

model = None
tokenizer = None
label_encoder = None

def load_artifacts():
    global model, tokenizer, label_encoder
    if os.path.exists(MODEL_FILE):
        try:
            model = keras.models.load_model(MODEL_FILE)
            print(f"Modelo cargado: {MODEL_FILE}")
        except Exception as e:
            print("Error cargando el modelo:", e)
            model = None
    else:
        print("No se encontró el archivo del modelo:", MODEL_FILE)

    if os.path.exists(TOKENIZER_FILE):
        with open(TOKENIZER_FILE, "rb") as f:
            tokenizer = pickle.load(f)
        print("Tokenizer cargado.")
    else:
        print("No se encontró tokenizer.pkl en ml_models/")

    if os.path.exists(LE_FILE):
        with open(LE_FILE, "rb") as f:
            label_encoder = pickle.load(f)
        print("LabelEncoder cargado.")
    else:
        print("No se encontró label_encoder.pkl en ml_models/")

load_artifacts()

import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\sáéíóúüñ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global model, tokenizer, label_encoder
    data = request.get_json() or {}
    lyrics = data.get("lyrics", "")
    if not lyrics:
        return jsonify({"error": "No se recibieron letras"}), 400

    if tokenizer is None or model is None or label_encoder is None:
        return jsonify({"error": "Model/tokenizer/label encoder no están cargados en el servidor"}), 500

    cleaned = clean_text(lyrics)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

    pred_probs = model.predict(padded)
    pred_probs = np.asarray(pred_probs).reshape(-1)
    if pred_probs.size == 1:
        prob_pos = float(pred_probs[0])
        pred_index = int(prob_pos > 0.5)

        try:
            pred_label = label_encoder.inverse_transform([pred_index])[0]
        except Exception:
            pred_label = str(pred_index)
        probs = {
            str(label_encoder.classes_[0]): float(1 - prob_pos),
            str(label_encoder.classes_[1]): float(prob_pos)
        } if hasattr(label_encoder, 'classes_') else {"0": 1 - prob_pos, "1": prob_pos}
    else:
        pred_index = int(np.argmax(pred_probs))
        try:
            pred_label = label_encoder.inverse_transform([pred_index])[0]
            class_names = list(label_encoder.classes_)
        except Exception:
            pred_label = str(pred_index)
            class_names = [str(i) for i in range(len(pred_probs))]
        probs = {c: float(p) for c, p in zip(class_names, pred_probs.tolist())}

    return jsonify({
        "prediction": pred_label,
        "pred_index": int(pred_index),
        "probs": probs
    })


@app.route("/reload_artifacts", methods=["POST"])
def reload_artifacts():
    load_artifacts()
    return jsonify({"status": "artifacts reloaded"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
