"""Train a handwritten digit classifier on MNIST.

Usage:
    python -m src.train            # full training (60k train / 10k test)
    python -m src.train --quick    # fast smoke run on a subset
"""
import argparse
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

from src.common import MODEL_PATH, OUTPUT_DIR, load_mnist


def build_model(max_iter: int = 30, seed: int = 42) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=128,
        learning_rate_init=1e-3,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=seed,
        verbose=True,
    )


def plot_confusion(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title("MNIST Test Set - Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return cm


def plot_samples(X, y_true, y_pred, path, n=20, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    fig, axes = plt.subplots(4, 5, figsize=(10, 8.5))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(X[i].reshape(28, 28), cmap="gray")
        ok = y_true[i] == y_pred[i]
        ax.set_title(f"pred {y_pred[i]} / true {y_true[i]}", color="green" if ok else "red", fontsize=10)
        ax.axis("off")
    fig.suptitle("Sample Predictions on MNIST Test Images", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_misclassified(X, y_true, y_pred, path, n=10):
    wrong = np.flatnonzero(y_true != y_pred)[:n]
    if len(wrong) == 0:
        return
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.6))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), wrong):
        ax.imshow(X[i].reshape(28, 28), cmap="gray")
        ax.set_title(f"pred {y_pred[i]} / true {y_true[i]}", color="red", fontsize=10)
    fig.suptitle("Examples of Misclassified Digits", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss(model, path):
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(model.loss_curve_, marker="o", label="training loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss")
    ax2 = ax1.twinx()
    ax2.plot(model.validation_scores_, marker="s", color="tab:orange", label="validation accuracy")
    ax2.set_ylabel("Validation accuracy")
    fig.legend(loc="center", bbox_to_anchor=(0.5, 0.55))
    ax1.set_title("Training Curve")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train MNIST digit classifier")
    parser.add_argument("--quick", action="store_true", help="train on 10k samples for a fast smoke test")
    parser.add_argument("--max-iter", type=int, default=30)
    args = parser.parse_args()

    print("Loading MNIST ...")
    X_train, X_test, y_train, y_test = load_mnist()
    if args.quick:
        X_train, y_train = X_train[:10000], y_train[:10000]
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    model = build_model(max_iter=args.max_iter)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    print(f"\nTraining time: {train_time:.1f}s")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {acc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    print(f"Saved model -> {MODEL_PATH}")

    cm = plot_confusion(y_test, y_pred, OUTPUT_DIR / "confusion_matrix.png")
    plot_samples(X_test, y_test, y_pred, OUTPUT_DIR / "sample_predictions.png")
    plot_misclassified(X_test, y_test, y_pred, OUTPUT_DIR / "misclassified.png")
    plot_loss(model, OUTPUT_DIR / "training_curve.png")

    metrics = {
        "model": "MLPClassifier(256,128) relu/adam",
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "epochs_run": int(model.n_iter_),
        "train_time_sec": round(train_time, 1),
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(acc, 4),
        "per_class": {str(k): {m: round(v, 4) for m, v in report[str(k)].items()} for k in range(10)},
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "confusion_matrix": cm.tolist(),
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved plots and metrics -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
