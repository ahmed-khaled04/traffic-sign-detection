"""
train.py — End-to-end training script for traffic sign recognition.

Usage:
    python train.py --data data/raw/archive --output results/
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import load_dataset, preprocess_batch
from features       import extract_features_batch
from classifier     import train, evaluate, save_model
from utils          import ensure_dir, save_json, GTSRB_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description="Train traffic sign SVM classifier")
    parser.add_argument("--data",   default="data/raw/archive", help="Path to GTSRB archive folder")
    parser.add_argument("--output", default="results/",         help="Directory to save model and results")
    return parser.parse_args()


def _build_features(images, labels, desc):
    print(f"  Preprocessing {desc}...")
    preprocessed = preprocess_batch(images)
    # Segmentation is skipped here: GTSRB images are already cropped to the
    # sign ROI, so color masking would destroy informative pixels (e.g. the
    # white interior of speed-limit signs) rather than remove background.
    print(f"  Extracting HOG features for {desc}...")
    X = extract_features_batch(preprocessed)
    y = np.array(labels)
    return X, y


def main():
    args = parse_args()
    ensure_dir(args.output)

    # 1. Load train and test splits from the provided CSVs
    print("Loading Train split...")
    train_images, train_labels = load_dataset(args.data, split="Train")
    print(f"  {len(train_images)} images across {len(set(train_labels))} classes")

    print("Loading Test split...")
    test_images, test_labels = load_dataset(args.data, split="Test")
    print(f"  {len(test_images)} images")

    # 2–3. Preprocess → HOG for both splits
    X_train, y_train = _build_features(train_images, train_labels, "train")
    X_test,  y_test  = _build_features(test_images,  test_labels,  "test")
    print(f"  Feature matrix — train: {X_train.shape}  test: {X_test.shape}")

    # 5. Train
    print("Training SVM (this may take a few minutes)...")
    model = train(X_train, y_train)

    # 6. Evaluate
    print("Evaluating...")
    results = evaluate(model, X_test, y_test)
    print(f"\n  Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print("\n" + results["report"])

    # 7. Save metrics JSON
    metrics_path = os.path.join(args.output, "metrics.json")
    save_json({"accuracy": results["accuracy"]}, metrics_path)
    print(f"  Metrics saved → {metrics_path}")

    # 8. Save confusion matrix PNG
    cm = results["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix — GTSRB SVM")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.tight_layout()
    cm_path = os.path.join(args.output, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=120)
    plt.close()
    print(f"  Confusion matrix saved → {cm_path}")

    # 9. Save model
    model_path = os.path.join(args.output, "svm_model.pkl")
    save_model(model, model_path)
    print(f"  Model saved → {model_path}")


if __name__ == "__main__":
    main()
