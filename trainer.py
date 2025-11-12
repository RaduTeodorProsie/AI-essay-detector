from collections import Counter
import math
import pandas as pd
import joblib
from sklearn import metrics
import argparse

def words_from_clean_text(s: str):
    """Return tokens by splitting the already-cleaned text on whitespace.
    Assumes text is already lowercased/normalized/punctuation-removed by you.
    """
    if s is None:
        return []
    return [w for w in str(s).split() if w]  # no extra processing

def train_multinomial_nb(docs, labels, alpha=1.0):
    """
    Train Multinomial Naive Bayes using space-separated already-cleaned docs.
    docs: iterable of cleaned text strings
    labels: iterable of class labels (ints or categories)
    returns a dict model with priors, word_counts, totals, V, alpha
    """
    classes = set(labels)
    doc_count = len(labels)

    class_doc_count = Counter(labels)
    class_word_counts = {c: Counter() for c in classes}
    class_total_words = {c: 0 for c in classes}
    vocab = set()

    for text, c in zip(docs, labels):
        tokens = words_from_clean_text(text)
        class_word_counts[c].update(tokens)
        class_total_words[c] += len(tokens)
        vocab.update(tokens)

    V = len(vocab)
    priors = {c: class_doc_count[c] / doc_count for c in classes}

    model = {
        "priors": priors,
        "word_counts": class_word_counts,
        "total_words": class_total_words,
        "vocab": vocab,
        "V": V,
        "alpha": alpha
    }
    return model

def predict_one(model, text):
    tokens = words_from_clean_text(text)
    counts = Counter(tokens)
    best_class = None
    best_log_prob = -math.inf
    alpha = model["alpha"]
    V = model["V"]

    for c, prior in model["priors"].items():

        log_prob = math.log(prior) if prior > 0 else -math.inf
        total_words_c = model["total_words"][c]
        wc = model["word_counts"][c]
        denom = total_words_c + alpha * V

        for word, cnt in counts.items():
            num = wc.get(word, 0) + alpha
            log_w_given_c = math.log(num / denom)
            log_prob += cnt * log_w_given_c

        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_class = c

    return best_class

def predict(model, docs):
    return [predict_one(model, d) for d in docs]

def evaluate(y_true, y_pred):
    print("Accuracy:", metrics.accuracy_score(y_true, y_pred))
    print("\nClassification report:\n", metrics.classification_report(y_true, y_pred, digits=4))
    print("\nConfusion matrix:\n", metrics.confusion_matrix(y_true, y_pred))

def main(args):
    train_path = args.train
    test_path = args.test
    text_col = args.text_col
    label_col = args.label_col
    alpha = args.alpha
    out_model = args.out_model

    # Load training CSV
    print(f"Loading training data from: {train_path}")
    df_train = pd.read_csv(train_path)
    print("Train columns:", df_train.columns.tolist())

    if text_col not in df_train.columns or label_col not in df_train.columns:
        raise SystemExit(f"Training CSV must contain '{text_col}' and '{label_col}' columns. Found: {df_train.columns.tolist()}")

    df_train = df_train.dropna(subset=[text_col, label_col]).reset_index(drop=True)
    print(f"Training rows after dropna: {len(df_train)}")

    # Convert labels to integer for training
    df_train[label_col] = pd.to_numeric(df_train[label_col], errors='coerce')
    if df_train[label_col].isna().any():
        raise SystemExit("Training label column contains values that could not be converted to numeric.")
    df_train[label_col] = df_train[label_col].astype(int)

    X_train = df_train[text_col].tolist()
    y_train = df_train[label_col].tolist()

    # Train model on the training CSV
    print("Training Multinomial NB on training dataset...")
    model = train_multinomial_nb(X_train, y_train, alpha=alpha)
    print(f"Trained model. Vocab size: {model['V']}  Alpha: {model['alpha']}")

    # Load test CSV
    print(f"\nLoading test data from: {test_path}")
    df_test = pd.read_csv(test_path)
    print("Test columns:", df_test.columns.tolist())

    if text_col not in df_test.columns or label_col not in df_test.columns:
        raise SystemExit(f"Test CSV must contain '{text_col}' and '{label_col}' columns. Found: {df_test.columns.tolist()}")

    df_test = df_test.dropna(subset=[text_col, label_col]).reset_index(drop=True)
    print(f"Test rows after dropna: {len(df_test)}")

    # Convert labels to integer for testing
    df_test[label_col] = pd.to_numeric(df_test[label_col], errors='coerce')
    if df_test[label_col].isna().any():
        raise SystemExit("Test label column contains values that could not be converted to numeric.")
    df_test[label_col] = df_test[label_col].astype(int)

    X_test = df_test[text_col].tolist()
    y_test = df_test[label_col].tolist()

    # Predict on the external test set (no retraining)
    print("Predicting on test dataset...")
    y_pred = predict(model, X_test)

    # Evaluate
    print("\n=== Evaluation on external test set ===")
    evaluate(y_test, y_pred)

    # Save model
    joblib.dump(model, out_model)
    print(f"\nModel saved to {out_model}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train on one CSV and evaluate on another (no extra tokenization).")
    p.add_argument("--train", default="data/ai-vs-human-text-clean.csv", help="Path to training CSV (default: data/ai-vs-human-text-clean.csv)")
    p.add_argument("--test", default="data/ai-vs-human-text-clean-testing.csv", help="Path to test CSV (default: data/ai-vs-human-text-clean-testing.csv)")
    p.add_argument("--text-col", default="text", help="Name of the text column (default: text)")
    p.add_argument("--label-col", default="generated", help="Name of the label column (default: generated)")
    p.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing alpha (default 1.0)")
    p.add_argument("--out-model", default="nb_scratch_model.joblib", help="File to save trained model")
    args = p.parse_args()
    main(args)
