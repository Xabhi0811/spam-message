import joblib
import numpy as np
import re

MODEL_PATH = "spam_detector_model.joblib"

def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+|www\S+", " url ", t)
    t = re.sub(r"[^a-z0-9\s₹!]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def w2v_embed(text, model, size=100):
    tokens = clean_text(text).split()
    vectors = [model.wv[t] for t in tokens if t in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(size)

def predict_spam(message: str):
    tfidf, w2v, scaler, svm = joblib.load(MODEL_PATH)

    clean = clean_text(message)
    tfidf_vec = tfidf.transform([clean]).toarray()[0]
    w2v_vec = w2v_embed(message, w2v)

    hybrid = np.concatenate([tfidf_vec, w2v_vec]).reshape(1, -1)
    hybrid = scaler.transform(hybrid)

    pred = svm.predict(hybrid)[0]
    return "Spam" if pred == 1 else "Ham"


if __name__ == "__main__":
    # quick CLI test
    while True:
        msg = input("\nMessage (or 'quit'): ")
        if msg.lower().strip() in ("quit", "exit"):
            break
        print("Prediction:", predict_spam(msg))
