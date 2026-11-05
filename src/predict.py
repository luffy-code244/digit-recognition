"""Predict handwritten digits with the trained model.

Usage:
    python -m src.predict --image path/to/digit.png   # predict a single image
    python -m src.predict --demo                      # demo on random MNIST test samples
"""
import argparse

import joblib
import numpy as np
from PIL import Image, ImageOps

from src.common import MODEL_PATH


def preprocess_image(path: str) -> np.ndarray:
    """Convert an arbitrary image into a 1x784 MNIST-style vector.

    Steps: grayscale -> invert if background is light -> crop to the digit ->
    fit into a 20x20 box keeping aspect ratio -> centre on a 28x28 canvas by
    centre of mass (like the original MNIST preprocessing) -> scale to [0, 1].
    """
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    if arr.mean() > 127:  # light background, dark ink -> invert to MNIST style
        img = ImageOps.invert(img)
        arr = np.asarray(img, dtype=np.float32)
    mask = arr > 50
    if mask.any():
        rows, cols = np.where(mask)
        img = img.crop((cols.min(), rows.min(), cols.max() + 1, rows.max() + 1))
    w, h = img.size
    scale = 20.0 / max(w, h)
    img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(img, ((28 - img.size[0]) // 2, (28 - img.size[1]) // 2))
    out = np.asarray(canvas, dtype=np.float32) / 255.0
    # shift so the centre of mass is at the centre
    total = out.sum()
    if total > 0:
        ys, xs = np.indices(out.shape)
        cy, cx = (ys * out).sum() / total, (xs * out).sum() / total
        out = np.roll(out, (int(round(14 - cy)), int(round(14 - cx))), axis=(0, 1))
    return out.reshape(1, -1)


def load_model(path=MODEL_PATH):
    if not path.exists():
        raise SystemExit(f"Model not found at {path}. Run `python -m src.train` first.")
    return joblib.load(path)


def predict_image(model, path: str):
    x = preprocess_image(path)
    probs = model.predict_proba(x)[0]
    digit = int(np.argmax(probs))
    print(f"{path}: predicted digit = {digit} (confidence {probs[digit]:.2%})")
    top3 = np.argsort(probs)[::-1][:3]
    print("  top-3:", ", ".join(f"{d} ({probs[d]:.1%})" for d in top3))
    return digit, probs


def demo(model, n: int = 10, seed: int = 7):
    from src.common import load_mnist

    _, X_test, _, y_test = load_mnist()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_test), size=n, replace=False)
    preds = model.predict(X_test[idx])
    correct = 0
    for i, p in zip(idx, preds):
        ok = p == y_test[i]
        correct += ok
        print(f"test #{i:5d}: true={y_test[i]} pred={p} {'OK' if ok else 'WRONG'}")
    print(f"{correct}/{n} correct")


def main():
    parser = argparse.ArgumentParser(description="Predict handwritten digits")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", nargs="+", help="path(s) to image file(s)")
    g.add_argument("--demo", action="store_true", help="run on random MNIST test samples")
    parser.add_argument("-n", type=int, default=10, help="number of demo samples")
    args = parser.parse_args()

    model = load_model()
    if args.demo:
        demo(model, n=args.n)
    else:
        for p in args.image:
            predict_image(model, p)


if __name__ == "__main__":
    main()
