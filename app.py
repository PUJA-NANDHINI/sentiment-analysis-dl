from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_web", methods=["POST"])
def predict_web():
    text = request.form["text"]

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)

    pred = model.predict(padded, verbose=0)[0][0]

    if pred > 0.7:
        sentiment = "Positive 😊"
    elif pred < 0.3:
        sentiment = "Negative 😞"
    else:
        sentiment = "Neutral 😐"

    return render_template("index.html", result=sentiment)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)