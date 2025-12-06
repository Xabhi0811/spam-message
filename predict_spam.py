import joblib
import numpy as np
import re

MODEL_PATH = "spam_detector_model.joblib"

def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+|www\S+", " url ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def w2v_embed(text, model, size=100):
    tokens = clean_text(text).split()
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(size)
    return np.mean(vectors, axis=0)

def predict_spam(message):
    tfidf, w2v_model, svm = joblib.load(MODEL_PATH)

    tfidf_vec = tfidf.transform([message]).toarray()[0]
    w2v_vec = w2v_embed(message, w2v_model)

    hybrid = np.concatenate([tfidf_vec, w2v_vec]).reshape(1, -1)

    pred = svm.predict(hybrid)[0]
    prob = svm.predict_proba(hybrid)[0]

    return ("Spam" if pred == 1 else "Ham"), prob[1], prob[0]
