"""
segmentation.py — Color-based region segmentation.

Accepts normalized HSV images and isolates sign-relevant regions
(red, blue, yellow) using HSV range thresholds and morphological cleanup.
"""

from typing import Tuple

import cv2
import numpy as np


# HSV ranges for sign colors.
# Note: OpenCV HSV uses H in [0,179], S and V in [0,255].
# When working with normalized [0,1] images we scale back for masking.
_RED_LOWER_1  = np.array([0,   0.40, 0.40], dtype=np.float32)
_RED_UPPER_1  = np.array([0.056, 1.0, 1.0], dtype=np.float32)   # hue  0–10 / 179
_RED_LOWER_2  = np.array([0.894, 0.40, 0.40], dtype=np.float32)
_RED_UPPER_2  = np.array([1.0,   1.0, 1.0],  dtype=np.float32)  # hue 160–179 / 179

_BLUE_LOWER   = np.array([0.559, 0.40, 0.20], dtype=np.float32) # hue 100–130 / 179
_BLUE_UPPER   = np.array([0.726, 1.0,  1.0],  dtype=np.float32)

_YELLOW_LOWER = np.array([0.084, 0.40, 0.40], dtype=np.float32) # hue 15–35 / 179
_YELLOW_UPPER = np.array([0.196, 1.0,  1.0],  dtype=np.float32)

_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def create_color_mask(hsv_image: np.ndarray) -> np.ndarray:
    """
    Build a binary mask covering red, blue, and yellow sign regions.

    Parameters
    ----------
    hsv_image : float32 ndarray (H, W, 3), HSV values normalized to [0, 1]

    Returns
    -------
    uint8 binary mask (H, W), 255 where sign colors are detected
    """
    red1   = cv2.inRange(hsv_image, _RED_LOWER_1,  _RED_UPPER_1)
    red2   = cv2.inRange(hsv_image, _RED_LOWER_2,  _RED_UPPER_2)
    blue   = cv2.inRange(hsv_image, _BLUE_LOWER,   _BLUE_UPPER)
    yellow = cv2.inRange(hsv_image, _YELLOW_LOWER, _YELLOW_UPPER)

    combined = cv2.bitwise_or(cv2.bitwise_or(red1, red2), cv2.bitwise_or(blue, yellow))
    cleaned  = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, _MORPH_KERNEL)
    return cleaned


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a binary mask to an image, zeroing out non-masked pixels."""
    return cv2.bitwise_and(image, image, mask=mask)


def segment_image(hsv_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment a preprocessed HSV image by its sign colors.

    Parameters
    ----------
    hsv_image : float32 ndarray (32, 32, 3), normalized HSV

    Returns
    -------
    segmented : float32 ndarray (32, 32, 3) — image with non-sign pixels zeroed
    mask      : uint8 ndarray (32, 32) — binary mask for visualization
    """
    mask      = create_color_mask(hsv_image)
    segmented = apply_mask(hsv_image, mask)
    return segmented, mask
