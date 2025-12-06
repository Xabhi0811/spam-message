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
    return jsonify({"status": "ok", "message": "Spam detection API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data is None or "message" not in data:
            return jsonify({"error": "Missing 'message' field"}), 400

        message = data["message"].strip()
        if message == "":
            return jsonify({"error": "Message must be non-empty"}), 400

        label = model.predict([message])[0]

        return jsonify({
            "message": message,
            "prediction": label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
