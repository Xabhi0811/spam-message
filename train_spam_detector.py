import pandas as pd
import numpy as np
import re
import joblib
import os

from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# -------------------------------
# TEXT CLEANER
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------------------------------
# WORD2VEC EMBEDDING FUNCTION
# -------------------------------
def w2v_embed(text, model, size=100):
    tokens = clean_text(text).split()
    vectors = [model.wv[word] for word in tokens if word in model.wv]

    if len(vectors) == 0:
        return np.zeros(size)

    return np.mean(vectors, axis=0)


# -------------------------------
# LOAD DATA
# -------------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        sep=",",
        usecols=[0, 1],
        names=["label", "message"],
        header=None,
        engine="python"
    )

    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()
    return df


# -------------------------------
# LABEL CLEANING
# -------------------------------
def preprocess_labels(df: pd.DataFrame):
    df["label_clean"] = df["label"].str.lower()

    label_map = {
        "ham": 0,
        "not spam": 0,
        "normal": 0,
        "0": 0,
        0: 0,
        "spam": 1,
        "1": 1,
        1: 1
    }

    df["label_num"] = df["label_clean"].map(label_map)
    df = df.dropna(subset=["label_num"])

    X = df["message"].values
    y = df["label_num"].astype(int).values
    return X, y


# -------------------------------
# EVALUATION
# -------------------------------
def evaluate_model(y_true, y_pred):
    print("=== Evaluation Metrics ===")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_true, y_pred):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Ham", "Spam"]))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


# -------------------------------
# MAIN TRAINING PIPELINE
# -------------------------------
def main():
    csv_path = "spam_dataset.csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    print("[INFO] Loading data...")
    df = load_data(csv_path)
    X, y = preprocess_labels(df)

    print(f"[INFO] Samples: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------------------------
    # TRAIN TF-IDF
    # ---------------------------------------
    print("[INFO] Training TF-IDF vectorizer...")

    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98,
    )

    tfidf.fit(X_train)

    # ---------------------------------------
    # TRAIN WORD2VEC
    # ---------------------------------------
    print("[INFO] Training Word2Vec model...")
    sentences = [clean_text(text).split() for text in X_train]

    w2v_model = Word2Vec(
        sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4
    )

    # ---------------------------------------
    # CREATE HYBRID FEATURES
    # ---------------------------------------
    print("[INFO] Creating hybrid (TF-IDF + W2V) feature vectors...")

    X_train_vec = []
    for msg in X_train:
        tfidf_vec = tfidf.transform([msg]).toarray()[0]
        w2v_vec = w2v_embed(msg, w2v_model)
        hybrid = np.concatenate([tfidf_vec, w2v_vec])
        X_train_vec.append(hybrid)

    X_train_vec = np.array(X_train_vec)

    X_test_vec = []
    for msg in X_test:
        tfidf_vec = tfidf.transform([msg]).toarray()[0]
        w2v_vec = w2v_embed(msg, w2v_model)
        hybrid = np.concatenate([tfidf_vec, w2v_vec])
        X_test_vec.append(hybrid)

    X_test_vec = np.array(X_test_vec)

    # ---------------------------------------
    # TRAIN SVM
    # ---------------------------------------
    print("[INFO] Training SVM classifier...")
    svm = SVC(kernel="linear", probability=True, class_weight="balanced")
    svm.fit(X_train_vec, y_train)

    # ---------------------------------------
    # EVALUATE
    # ---------------------------------------
    print("\n[INFO] Evaluating model...")
    y_pred = svm.predict(X_test_vec)
    evaluate_model(y_test, y_pred)

    # ---------------------------------------
    # SAVE MODEL
    # ---------------------------------------
    joblib.dump((tfidf, w2v_model, svm), "spam_detector_model.joblib")
    print("[SUCCESS] Model saved to spam_detector_model.joblib")


if __name__ == "__main__":
    main()
