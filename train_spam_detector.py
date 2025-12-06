import pandas as pd
import numpy as np
import re
import joblib
import os

from gensim.models import Word2Vec
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


# -------------------------------
# TEXT CLEANER
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"[^a-z0-9\s₹!]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------------------------------
# WORD2VEC EMBEDDING
# -------------------------------
def w2v_embed(text, model, size=100):
    tokens = clean_text(text).split()
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(size)


# -------------------------------
# LOAD CSV
# -------------------------------
def load_data(csv_path: str):
    df = pd.read_csv(
        csv_path, sep=",", usecols=[0, 1],
        names=["label", "message"],
        header=None, engine="python"
    )
    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].str.lower().str.strip()
    df["message"] = df["message"].astype(str).str.strip()
    return df


# -------------------------------
# LABEL PROCESSING
# -------------------------------
def preprocess_labels(df):
    mapping = {"ham": 0, "not spam": 0, "normal": 0, "spam": 1}
    df["label_num"] = df["label"].map(mapping)
    df = df.dropna(subset=["label_num"])
    X = df["message"].values
    y = df["label_num"].values.astype(int)
    return X, y


# -------------------------------
# EVALUATION
# -------------------------------
def evaluate_model(y_true, y_pred):
    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Ham", "Spam"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


# -------------------------------
# MAIN PIPELINE
# -------------------------------
def main():
    csv_path = "spam_dataset.csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    print("[INFO] Loading dataset...")
    df = load_data(csv_path)
    X, y = preprocess_labels(df)

    print(f"[INFO] Samples loaded: {len(X)}")
    print(f"[INFO] Ham: {(y == 0).sum()}, Spam: {(y == 1).sum()}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------------------------
    # TF-IDF (train on CLEAN text)
    # ---------------------------------------
    print("[INFO] Training TF-IDF...")
    X_train_clean = [clean_text(t) for t in X_train]

    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=6000
    )
    tfidf.fit(X_train_clean)

    # ---------------------------------------
    # TRAIN WORD2VEC
    # ---------------------------------------
    print("[INFO] Training Word2Vec...")
    sentences = [clean_text(t).split() for t in X_train]
    w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

    # ---------------------------------------
    # COMBINE FEATURES (TF-IDF + W2V)
    # ---------------------------------------
    def hybrid(text):
        clean = clean_text(text)
        tfidf_vec = tfidf.transform([clean]).toarray()[0]
        w2v_vec = w2v_embed(text, w2v)
        return np.concatenate([tfidf_vec, w2v_vec])

    print("[INFO] Generating vectors...")
    X_train_vec = np.array([hybrid(t) for t in X_train])
    X_test_vec = np.array([hybrid(t) for t in X_test])

    # Normalize features (optional but usually helps)
    scaler = StandardScaler()
    X_train_vec = scaler.fit_transform(X_train_vec)
    X_test_vec = scaler.transform(X_test_vec)

    # ---------------------------------------
    # TRAIN LinearSVC
    # ---------------------------------------
    print("[INFO] Training LinearSVC...")
    svm = LinearSVC(class_weight="balanced")
    svm.fit(X_train_vec, y_train)

    # ---------------------------------------
    # EVALUATE
    # ---------------------------------------
    y_pred = svm.predict(X_test_vec)
    evaluate_model(y_test, y_pred)

    # ---------------------------------------
    # SAVE MODEL
    # ---------------------------------------
    joblib.dump((tfidf, w2v, scaler, svm), "spam_detector_model.joblib")
    print("\n[SUCCESS] Model saved: spam_detector_model.joblib")


if __name__ == "__main__":
    main()
