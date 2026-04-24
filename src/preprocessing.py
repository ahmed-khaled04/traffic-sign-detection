"""
preprocessing.py — Image loading and preprocessing pipeline.

Loads raw GTSRB images from disk, crops to ROI, resizes to 32×32,
converts BGR→HSV, and normalizes to float32 [0, 1].
"""

import os
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd


def load_dataset(data_dir: str, split: str = "Train") -> Tuple[List[np.ndarray], List[int]]:
    """
    Load images from the Kaggle GTSRB archive layout.

    Expects structure:
        data_dir/Train.csv  (or Test.csv)
        data_dir/Train/<class_id>/*.png
        data_dir/Test/*.png

    CSV columns (comma-separated):
        Width, Height, Roi.X1, Roi.Y1, Roi.X2, Roi.Y2, ClassId, Path

    The Path column is relative to data_dir (e.g. "Train/0/00000_00000_00000.png").
    Images are cropped to the ROI bounding box before being returned.

    Parameters
    ----------
    data_dir : path to the archive/ folder  (e.g. "data/raw/archive")
    split    : "Train" or "Test"

    Returns
    -------
    images : list of BGR uint8 arrays, variable size (cropped to ROI)
    labels : list of int class IDs (0–42)
    """
    csv_path = os.path.join(data_dir, f"{split}.csv")
    df = pd.read_csv(csv_path)

    images: List[np.ndarray] = []
    labels: List[int] = []

    for _, row in df.iterrows():
        img_path = os.path.join(data_dir, row["Path"])
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Crop to ROI bounding box
        x1, y1, x2, y2 = int(row["Roi.X1"]), int(row["Roi.Y1"]), int(row["Roi.X2"]), int(row["Roi.Y2"])
        img = img[y1:y2, x1:x2]
        if img.size == 0:
            continue

        images.append(img)
        labels.append(int(row["ClassId"]))

    return images, labels


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Resize a single BGR image to 32×32, convert to HSV, normalize to [0, 1].

    Parameters
    ----------
    image : BGR uint8 ndarray of any size

    Returns
    -------
    float32 ndarray of shape (32, 32, 3) in HSV color space, values in [0, 1]
    """
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    normalized = hsv.astype(np.float32) / np.array([179.0, 255.0, 255.0], dtype=np.float32)
    return normalized


def preprocess_batch(images: List[np.ndarray]) -> np.ndarray:
    """
    Apply preprocess_image to a list of images.

    Returns
    -------
    float32 ndarray of shape (N, 32, 32, 3)
    """
    return np.stack([preprocess_image(img) for img in images], axis=0)
