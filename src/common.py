"""Shared helpers for loading and preprocessing MNIST."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "digit_model.joblib"
OUTPUT_DIR = ROOT / "outputs"
DATA_CACHE = ROOT / ".cache"


def load_mnist():
    """Download (or load cached) MNIST and return the standard 60k/10k split.

    Pixels are scaled to [0, 1] as float32.
    """
    cache = DATA_CACHE / "mnist.npz"
    if cache.exists():
        d = np.load(cache)
        X, y = d["X"], d["y"]
    else:
        from sklearn.datasets import fetch_openml

        X, y = fetch_openml(
            "mnist_784", version=1, as_frame=False, parser="auto",
            return_X_y=True, data_home=str(DATA_CACHE),
        )
        X = X.astype(np.uint8)
        y = y.astype(np.int64)
        DATA_CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, y=y)
    X = (X / 255.0).astype(np.float32)
    y = y.astype(np.int64)
    return X[:60000], X[60000:], y[:60000], y[60000:]
