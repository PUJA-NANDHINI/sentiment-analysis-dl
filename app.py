from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

@app.route("/")
def home():
    return "Sentiment Analysis API Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)

    pred = model.predict(padded)[0][0]

    if pred > 0.7:
        sentiment = "Positive 😊"
    elif pred < 0.3:
        sentiment = "Negative 😞"
    else:
        sentiment = "Neutral 😐"

    return jsonify({"sentiment": sentiment})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)