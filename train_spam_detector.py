# train_spam_detector.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
import joblib
import os

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # ✅ Correct column validation for YOUR dataset
    if "Category" not in df.columns or "Message" not in df.columns:
        raise ValueError("CSV must contain 'Category' and 'Message' columns.")

    # ✅ Clean missing values
    df = df.dropna(subset=["Category", "Message"])
    return df



def preprocess_labels(df: pd.DataFrame):
    df["label_num"] = df["Category"].map({"ham": 0, "spam": 1})

    if df["label_num"].isna().any():
        raise ValueError("Category column must contain only 'ham' or 'spam' values.")

    X = df["Message"].values
    y = df["label_num"].values
    return X, y



def build_pipeline() -> Pipeline:
    """
    Create a sklearn Pipeline: TF-IDF Vectorizer + Naive Bayes classifier.
    """
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",  # remove common English words
                    ngram_range=(1, 2),   # unigrams + bigrams
                    max_df=0.95,          # ignore very frequent words
                    min_df=2,             # ignore rare words
                ),
            ),
            ("clf", MultinomialNB()),
        ]
    )
    return pipeline


def evaluate_model(y_true, y_pred):
    """
    Print evaluation metrics.
    """
    print("=== Evaluation Metrics ===")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Ham", "Spam"]))

    print("\nConfusion Matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_true, y_pred))


def main():
    # 1. Set your dataset path here
    csv_path = "spam_dataset.csv"  # <- change this to your actual file

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset file not found at {csv_path}. Please check the path."
        )

    print("[INFO] Loading data...")
    df = load_data(csv_path)
    X, y = preprocess_labels(df)


    print(f"[INFO] Total samples: {len(X)}")

    # 2. Train / validation / test split
    # First: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    # Second: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
    )
    # 0.85 * 0.1765 ≈ 0.15 → so overall: 70% train, 15% val, 15% test

    print(f"[INFO] Train size: {len(X_train)}")
    print(f"[INFO] Val size  : {len(X_val)}")
    print(f"[INFO] Test size : {len(X_test)}")

    # 3. Build model pipeline
    print("[INFO] Building model pipeline...")
    model = build_pipeline()

    # 4. Train model
    print("[INFO] Training model...")
    model.fit(X_train, y_train)

    # 5. Validation evaluation
    print("\n[INFO] Evaluating on validation set...")
    y_val_pred = model.predict(X_val)
    evaluate_model(y_val, y_val_pred)

    # 6. Final evaluation on test set
    print("\n[INFO] Evaluating on test set...")
    y_test_pred = model.predict(X_test)
    evaluate_model(y_test, y_test_pred)

    # 7. Save trained model (pipeline contains both TF-IDF + classifier)
    model_path = "spam_detector_model.joblib"
    joblib.dump(model, model_path)
    print(f"\n[SUCCESS] Model saved to {model_path}")


if __name__ == "__main__":
    main()
