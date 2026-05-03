"""
features.py — HOG + color histogram feature extraction.

Converts HSV images to grayscale for HOG and also computes per-channel
HSV histograms. Both are concatenated into a single feature vector.
"""

from typing import List, Tuple

import cv2
import numpy as np
from skimage.feature import hog


_HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    transform_sqrt=True,
)

_COLOR_BINS = 32


def _to_gray(hsv_image: np.ndarray) -> np.ndarray:
    """Convert a normalized float32 HSV image to a uint8 grayscale image."""
    uint8_hsv = (hsv_image * np.array([179.0, 255.0, 255.0])).astype(np.uint8)
    bgr = cv2.cvtColor(uint8_hsv, cv2.COLOR_HSV2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _color_histogram(hsv_image: np.ndarray) -> np.ndarray:
    """Normalized concatenated histogram over H, S, V channels."""
    h = np.histogram(hsv_image[:, :, 0], bins=_COLOR_BINS, range=(0.0, 1.0))[0]
    s = np.histogram(hsv_image[:, :, 1], bins=_COLOR_BINS, range=(0.0, 1.0))[0]
    v = np.histogram(hsv_image[:, :, 2], bins=_COLOR_BINS, range=(0.0, 1.0))[0]
    hist = np.concatenate([h, s, v]).astype(np.float64)
    return hist / (hist.sum() + 1e-7)


def extract_hog(image: np.ndarray) -> np.ndarray:
    """
    Compute HOG + color histogram feature vector for one image.

    Parameters
    ----------
    image : float32 ndarray (64, 64, 3), normalized HSV

    Returns
    -------
    1-D float64 ndarray — HOG features concatenated with color histogram
    """
    gray = _to_gray(image)
    hog_features = hog(gray, **_HOG_PARAMS, visualize=False)
    color_features = _color_histogram(image)
    return np.concatenate([hog_features, color_features])


def extract_hog_visual(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute HOG + color histogram features and return the HOG visualization.

    Returns
    -------
    features  : 1-D float64 ndarray
    hog_image : float64 ndarray (64, 64) suitable for display
    """
    gray = _to_gray(image)
    hog_features, hog_image = hog(gray, **_HOG_PARAMS, visualize=True)
    color_features = _color_histogram(image)
    return np.concatenate([hog_features, color_features]), hog_image


def extract_features_batch(images: List[np.ndarray]) -> np.ndarray:
    """
    Extract HOG + color histogram features for a list of images.

    Returns
    -------
    float64 ndarray of shape (N, D)
    """
    return np.stack([extract_hog(img) for img in images], axis=0)
