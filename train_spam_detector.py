import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib
import os


# ✅ LOAD DATA SAFELY (handles extra commas properly)
def load_data(csv_path: str) -> pd.DataFrame:
    # ✅ Read only first 2 columns, FORCE correct structure
    df = pd.read_csv(
        csv_path,
        sep=",",
        usecols=[0, 1],
        names=["label", "message"],
        header=None,          # ✅ THIS IS THE CRITICAL FIX
        engine="python"
    )

    # ✅ Drop broken rows
    df = df.dropna(subset=["label", "message"])

    # ✅ Clean spaces and quotes
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()

    print("[INFO] CSV columns detected:", df.columns.tolist())
    print("[INFO] First 5 labels:", df["label"].head().tolist())

    return df



# ✅ CLEAN & MAP LABELS SAFELY (works with ham/spam, 0/1, etc.)
def preprocess_labels(df: pd.DataFrame):
    df["label_clean"] = df["label"].astype(str).str.strip().str.lower()

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

    before = len(df)
    df = df.dropna(subset=["label_num"])
    after = len(df)

    print(f"[INFO] Dropped {before - after} rows with invalid labels")

    X = df["message"].astype(str).values
    y = df["label_num"].astype(int).values
    return X, y


# ✅ STRONG ML PIPELINE (TF-IDF + Logistic Regression)
def build_pipeline() -> Pipeline:
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_df=0.98,
                    min_df=1,
                    sublinear_tf=True
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=200,
                    solver="liblinear"
                ),
            ),
        ]
    )
    return pipeline


# ✅ EVALUATION METRICS
def evaluate_model(y_true, y_pred):
    print("=== Evaluation Metrics ===")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_true, y_pred):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Ham", "Spam"]))

    print("\nConfusion Matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_true, y_pred))


# ✅ MAIN TRAINING PIPELINE
def main():
    csv_path = "spam_dataset.csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at {csv_path}")

    print("[INFO] Loading data...")
    df = load_data(csv_path)

    X, y = preprocess_labels(df)
    print(f"[INFO] Total samples after cleaning: {len(X)}")

    # ✅ Safety check
    if len(X) < 100:
        raise ValueError("❌ Dataset too small after cleaning. CSV format still broken.")

    # ✅ TRAIN / VAL / TEST SPLIT
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
    )

    print(f"[INFO] Train size: {len(X_train)}")
    print(f"[INFO] Val size  : {len(X_val)}")
    print(f"[INFO] Test size : {len(X_test)}")

    # ✅ BUILD + TRAIN MODEL
    print("[INFO] Building model pipeline...")
    model = build_pipeline()

    print("[INFO] Training model...")
    model.fit(X_train, y_train)

    # ✅ VALIDATION
    print("\n[INFO] Evaluating on validation set...")
    y_val_pred = model.predict(X_val)
    evaluate_model(y_val, y_val_pred)

    # ✅ TEST
    print("\n[INFO] Evaluating on test set...")
    y_test_pred = model.predict(X_test)
    evaluate_model(y_test, y_test_pred)

    # ✅ SAVE MODEL
    model_path = "spam_detector_model.joblib"
    joblib.dump(model, model_path)
    print(f"\n[SUCCESS] Model saved to {model_path}")


if __name__ == "__main__":
    main()
