"""
features.py — HOG feature extraction.

Converts segmented HSV images to grayscale and computes Histogram of
Oriented Gradients (HOG) descriptors for use as classifier input.
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


def _to_gray(hsv_image: np.ndarray) -> np.ndarray:
    """Convert a normalized float32 HSV image to a uint8 grayscale image."""
    uint8_hsv = (hsv_image * np.array([179.0, 255.0, 255.0])).astype(np.uint8)
    bgr = cv2.cvtColor(uint8_hsv, cv2.COLOR_HSV2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def extract_hog(image: np.ndarray) -> np.ndarray:
    """
    Compute a flat HOG feature vector for one image.

    Parameters
    ----------
    image : float32 ndarray (32, 32, 3), normalized HSV (segmented)

    Returns
    -------
    1-D float64 ndarray of HOG features
    """
    gray = _to_gray(image)
    features = hog(gray, **_HOG_PARAMS, visualize=False)
    return features


def extract_hog_visual(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute HOG features and return the visualization image alongside them.

    Returns
    -------
    features  : 1-D float64 ndarray
    hog_image : float64 ndarray (32, 32) suitable for display
    """
    gray = _to_gray(image)
    features, hog_image = hog(gray, **_HOG_PARAMS, visualize=True)
    return features, hog_image


def extract_features_batch(images: List[np.ndarray]) -> np.ndarray:
    """
    Extract HOG features for a list of images.

    Returns
    -------
    float64 ndarray of shape (N, D) where D is the HOG vector length
    """
    return np.stack([extract_hog(img) for img in images], axis=0)
