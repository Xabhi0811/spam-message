from flask import Flask, request, jsonify
from flask_cors import CORS
from predict_spam import predict_spam

app = Flask(__name__)

# Allow all origins (frontend 3000 → backend 5000)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        # Preflight request response
        response = app.make_default_options_response()
        headers = response.headers

        headers["Access-Control-Allow-Origin"] = "*"
        headers["Access-Control-Allow-Headers"] = "Content-Type"
        headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"

        return response

    data = request.get_json()
    message = data.get("message", "")
    result = predict_spam(message)

    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(debug=True)
