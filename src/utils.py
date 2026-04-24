"""
utils.py — Shared constants and helpers used across all pipeline modules.
"""

import json
import os

GTSRB_CLASSES = {
    0:  "Speed limit (20km/h)",
    1:  "Speed limit (30km/h)",
    2:  "Speed limit (50km/h)",
    3:  "Speed limit (60km/h)",
    4:  "Speed limit (70km/h)",
    5:  "Speed limit (80km/h)",
    6:  "End of speed limit (80km/h)",
    7:  "Speed limit (100km/h)",
    8:  "Speed limit (120km/h)",
    9:  "No passing",
    10: "No passing (vehicles > 3.5t)",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles > 3.5t prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve (left)",
    20: "Dangerous curve (right)",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows (right)",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all restrictions",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing (> 3.5t)",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(data: dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    # Convert numpy types to native Python for JSON serialization
    def _convert(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return obj

    serializable = {k: _convert(v) for k, v in data.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
