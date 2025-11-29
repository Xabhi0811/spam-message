# predict_spam.py

import joblib
import os


MODEL_PATH = "spam_detector_model.joblib"


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. "
            "Train the model first by running: python train_spam_detector.py"
        )
    return joblib.load(MODEL_PATH)


def predict_spam(spam: str):
    model = load_model()
    # model is a Pipeline: TF-IDF + classifier
    pred = model.predict([spam])[0]
    prob = model.predict_proba([spam])[0]

    ham = "Spam" if pred == 1 else "Ham"
    spam_prob = prob[1]
    ham_prob = prob[0]

    return ham, spam_prob, ham_prob


if __name__ == "__main__":
    print("=== Spam Detection CLI ===")
    print("Type a spam (or 'quit' to exit):")
    model = load_model()

    while True:
        msg = input("\nspam: ")
        if msg.lower().strip() in ["quit", "exit"]:
            print("Goodbye!")
            break

        pred = model.predict([msg])[0]
        prob = model.predict_proba([msg])[0]

        ham = "Spam" if pred == 1 else "Ham"
        print(f"Prediction: {ham}")
        print(f"Probabilities -> Ham: {prob[0]:.4f}, Spam: {prob[1]:.4f}")
