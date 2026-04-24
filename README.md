# Traffic Sign Recognition

A computer vision pipeline for classifying German traffic signs using the [GTSRB dataset](https://benchmark.ini.rub.de/gtsrb_news.html). Built with classical CV techniques: HOG feature extraction and an SVM classifier — no deep learning.

**Test accuracy: ~90.2%** across 43 classes.

## Pipeline

```
Load Data → Preprocessing → Segmentation → Feature Extraction → Classification
```

| Step | Module | Description |
|---|---|---|
| Load Data | `train.py` | Read PPM images + ROI annotations from GTSRB CSV |
| Preprocessing | `src/preprocessing.py` | Crop ROI, resize to 32×32, normalize to [0,1], BGR→HSV |
| Segmentation | `src/segmentation.py` | HSV color masking (red / blue / yellow) |
| Feature Extraction | `src/features.py` | HOG — 9 orientations, 8×8 cells, 2×2 blocks, L2-Hys norm |
| Classification | `src/classifier.py` | StandardScaler → SVM (RBF kernel, C=10, γ=0.001) |

## Project Structure

```
traffic_sign_recognition/
├── data/
│   ├── raw/          # Original GTSRB images (not tracked in git)
│   └── processed/    # Cached feature arrays (optional)
├── src/
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── features.py
│   ├── classifier.py
│   └── utils.py      # Class names (0–42), JSON save, dir helpers
├── notebooks/        # Exploratory notebooks
├── results/          # svm_model.pkl, metrics.json, confusion_matrix.png
├── train.py          # CLI training script
├── app.py            # Streamlit demo app
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Place the GTSRB dataset at `data/raw/archive/`:

```
data/raw/archive/
├── Train.csv
├── Test.csv
├── Train/
│   ├── 0/  *.png
│   ├── 1/  *.png
│   ...
│   └── 42/ *.png
└── Test/   *.png
```

## Usage

**Train the model:**

```bash
python train.py --data data/raw/archive --output results/
```

Outputs saved to `results/`:
- `svm_model.pkl` — trained model
- `metrics.json` — test accuracy
- `confusion_matrix.png` — per-class confusion matrix

**Run the interactive demo:**

```bash
streamlit run app.py
```

Upload any traffic sign image and see each pipeline step visualized.

## Stack

- **OpenCV** — image I/O, color conversion, morphology
- **scikit-learn** — SVM, StandardScaler, metrics
- **scikit-image** — HOG computation and visualization
- **NumPy / pandas** — array ops, CSV parsing
- **Matplotlib** — confusion matrix plot
- **Streamlit** — demo web app
- **joblib** — model serialization
