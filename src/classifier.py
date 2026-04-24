"""
classifier.py — SVM-based traffic sign classifier.

Wraps a StandardScaler + SVC pipeline with train / evaluate / save /
load / predict helpers.
"""

from typing import Tuple

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def build_pipeline() -> Pipeline:
    """Return an unfitted StandardScaler → SVC(RBF) pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=10, gamma=0.001, probability=True, random_state=42)),
    ])


def train(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """
    Fit a new pipeline on training data and return it.

    Parameters
    ----------
    X_train : (N, D) feature matrix
    y_train : (N,) integer class labels
    """
    model = build_pipeline()
    model.fit(X_train, y_train)
    return model


def evaluate(model: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate a fitted model on test data.

    Returns
    -------
    dict with keys:
        accuracy         : float
        report           : str  (sklearn classification_report)
        confusion_matrix : np.ndarray
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy":         accuracy_score(y_test, y_pred),
        "report":           classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def save_model(model: Pipeline, path: str) -> None:
    """Persist a fitted pipeline to disk with joblib."""
    joblib.dump(model, path)


def load_model(path: str) -> Pipeline:
    """Load a previously saved pipeline from disk."""
    return joblib.load(path)


def predict(model: Pipeline, feature_vector: np.ndarray) -> Tuple[int, float]:
    """
    Classify a single feature vector.

    Parameters
    ----------
    feature_vector : 1-D ndarray (D,)

    Returns
    -------
    class_id    : int
    confidence  : float (probability of predicted class)
    """
    X = feature_vector.reshape(1, -1)
    class_id    = int(model.predict(X)[0])
    confidence  = float(model.predict_proba(X)[0, class_id])
    return class_id, confidence
