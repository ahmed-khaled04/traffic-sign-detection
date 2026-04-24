# Traffic Sign Recognition

Computer Vision course project — Traffic Sign Recognition using the GTSRB (German Traffic Sign Recognition Benchmark) dataset.

## Pipeline

```
Load Data → Preprocessing → Segmentation → Feature Extraction → Classification
```

1. **Load Data** — Read raw PPM images and ROI annotations from the GTSRB folder structure.
2. **Preprocessing** (`src/preprocessing.py`) — Crop to ROI, resize to 32×32, normalize to [0,1], convert BGR→HSV.
3. **Segmentation** (`src/segmentation.py`) — HSV color masking (red/blue/yellow) to isolate sign regions.
4. **Feature Extraction** (`src/features.py`) — HOG descriptors (9 orientations, 8×8 cells, 2×2 blocks, L2-Hys norm).
5. **Classification** (`src/classifier.py`) — StandardScaler → SVM (RBF, C=10, γ=0.001).

## Stack

- Python 3.x
- OpenCV (`opencv-python`) — image I/O, color conversion, morphology
- NumPy — array operations
- scikit-learn — scaler, SVM, metrics
- scikit-image — HOG feature visualization
- pandas — GTSRB CSV annotation parsing
- Matplotlib — confusion matrix plot
- Streamlit — demo web app
- joblib — model serialization

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `src/preprocessing.py` | Load GTSRB, crop ROI, resize, normalize, BGR→HSV |
| `src/segmentation.py` | HSV color masking for sign regions |
| `src/features.py` | HOG feature extraction + visualization |
| `src/classifier.py` | Train / evaluate / save / load / predict (SVM) |
| `src/utils.py` | GTSRB class names (0–42), JSON save, dir helper |

## Entry Points

| Script | Purpose |
|---|---|
| `train.py` | End-to-end training pipeline, saves model + results |
| `app.py` | Streamlit demo app — upload image, see each pipeline step |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Place GTSRB dataset at:
#   data/raw/archive/Train.csv + Train/ + Test.csv + Test/

# Train
python train.py --data data/raw/archive --output results/

# Run demo app
streamlit run app.py
```

## GTSRB Dataset Structure

```
data/raw/archive/
├── Train.csv          ← comma-separated: Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path
├── Test.csv
├── Train/
│   ├── 0/  *.png
│   ├── 1/  *.png
│   ...
│   └── 42/ *.png
├── Test/   *.png (flat)
└── Meta/   *.png (class icon images)
```

## Directory Layout

```
traffic_sign_recognition/
├── data/
│   ├── raw/          # Original GTSRB images (unmodified)
│   └── processed/    # (optional) cached feature arrays
├── src/              # Pipeline modules
├── notebooks/        # Exploratory notebooks
├── results/          # svm_model.pkl, metrics.json, confusion_matrix.png
├── train.py          # CLI training script
├── app.py            # Streamlit demo app
├── requirements.txt
└── CLAUDE.md
```

## Status

Pipeline fully implemented. Drop in the GTSRB dataset and run `train.py` to produce a trained model, then `streamlit run app.py` to launch the demo.
