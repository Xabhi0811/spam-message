# spam_api.py

from flask import Flask, request, jsonify
import joblib
import os
from flask_cors import CORS


MODEL_PATH = "spam_detector_model.joblib"

app = Flask(__name__)
CORS(app)


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. "
            "Train the model first by running: python train_spam_detector.py"
        )
    return joblib.load(MODEL_PATH)


model = load_model()


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "spam": "Spam detection API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({"error": "No JSON received"}), 400

        if "message" not in data:
            return jsonify({"error": "Missing 'message' field"}), 400

        message = data["message"]

        if not isinstance(message, str) or message.strip() == "":
            return jsonify({"error": "Message must be a non-empty string"}), 400

        pred = model.predict([message])[0]
        prob = model.predict_proba([message])[0]

        label = "Spam" if pred == 1 else "Ham"

        return jsonify({
            "input_message": message,
            "prediction": label,
            "probabilities": {
                "ham": float(prob[0]),
                "spam": float(prob[1])
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # For dev: do NOT use this in production directly
    app.run(host="0.0.0.0", port=5000, debug=True)
