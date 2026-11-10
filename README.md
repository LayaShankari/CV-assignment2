# Low-Resolution Image Classification: Nearest Neighbor vs. Histogram Classifier

A Computer Vision project comparing two traditional image classification algorithms—**1-Nearest Neighbor (1-NN)** and **Histogram Template Matching**—on low-resolution ($14 \times 14$ pixel) clothing images from the **Fashion-MNIST** benchmark dataset.

---

## 📌 Project Overview

Image classification traditionally relies on extracting features from raw pixels before passing them to a decision rule. This project investigates the importance of **spatial structure** versus **statistical pixel distribution** under extreme low-resolution constraints:

- **1-Nearest Neighbor (1-NN)** preserves spatial geometry by treating each pixel's exact coordinate as an independent feature.
- **Histogram Classifier** aggregates pixel intensities into a global frequency distribution, discarding all spatial information.
- Both methods are evaluated and compared on identical training and test splits, with results presented through an interactive **Streamlit web application**.

---

## 👕 Dataset: Fashion-MNIST

- **Source**: Zalando's Fashion-MNIST dataset of grayscale clothing items.
- **Classes (10 Categories)**:
  `0: T-shirt/top`, `1: Trouser`, `2: Pullover`, `3: Dress`, `4: Coat`, `5: Sandal`, `6: Shirt`, `7: Sneaker`, `8: Bag`, `9: Ankle boot`.
- **Resolution**: Subsampled from $28 \times 28$ to **$14 \times 14$ pixels** (196 features per image).
- **Normalization**: Pixel values are scaled from $[0, 255]$ to the range $[0.0, 1.0]$.
- **Format**: Foreground clothing items are bright ($>0.0$) against a dark background ($0.0$).

---

## 🔬 Algorithms & Methodology

### 1. 1-Nearest Neighbor (Pixel-Based Classification)
- **Concept**: Assigns a test sample the class label of the most similar training sample in high-dimensional feature space.
- **Feature Vector**: Flattens the $14 \times 14$ image into a $196$-dimensional vector $\mathbf{x} = [x_1, x_2, \dots, x_{196}]$.
- **Distance Metric**: Euclidean distance:
  $$d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{196} (x_i - y_i)^2}$$
- **Characteristics**: Sensitive to spatial alignment and silhouette shape (e.g. trouser legs, collars, soles).

### 2. Histogram Classifier (Intensity-Based Classification)
- **Concept**: Compares the distribution of light and dark pixels against average category templates.
- **Feature Extraction**: Constructs a normalized $16$-bin grayscale intensity histogram over the range $[0.0, 1.0]$:
  $$h_k = \frac{\text{count}(x \in \text{bin}_k)}{\sum h_j}$$
- **Training**: Computes the mean histogram vector $\mathbf{T}_c$ for each category $c \in \{0, \dots, 9\}$.
- **Distance Metric**: Chi-Square distance:
  $$\chi^2(\mathbf{P}, \mathbf{Q}) = \frac{1}{2} \sum_{k=1}^{16} \frac{(P_k - Q_k)^2}{P_k + Q_k + \epsilon}$$
- **Characteristics**: Invariant to pixel position and spatial shifts, but cannot differentiate distinct shapes that share similar overall brightness.

---

## 📊 Experimental Results & Key Insights

| Classifier | Feature Representation | Test Accuracy | Inference Speed | Spatial Awareness |
| :--- | :--- | :---: | :---: | :---: |
| **1-Nearest Neighbor** | Raw pixel vector ($196$-dim) | **~78% – 82%** | Moderate (scans dataset) | **Yes** (strict pixel alignment) |
| **Histogram Classifier** | Intensity histogram ($16$-bin) | **~25% – 30%** | Very Fast (10 templates) | **No** (purely statistical) |

### Key Takeaway:
Spatial layout is fundamental for shape-based visual recognition. While the histogram method is computationally lightweight and shift-invariant, global intensity counts alone cannot distinguish items with similar surface areas (such as a folded shirt vs. a bag). The 1-NN classifier significantly outperforms the histogram classifier because it evaluates exact coordinate patterns.

---

## 📁 Repository Structure

```text
cv-assign2/
│
├── app.py                     # Interactive Streamlit Web Application
├── requirements.txt           # Project Python dependencies
├── README.md                  # Project documentation
│
├── src/
│   ├── __init__.py            # Package initialization
│   ├── data.py                # Dataset loader (with local offline support)
│   ├── preprocessing.py       # Subsampling (14x14), normalization, and histogram extraction
│   ├── nearest_neighbor.py    # 1-NN training, inference, and closest-neighbor lookup
│   ├── histogram_classifier.py# Template building, Chi-Square distance calculation
│   └── evaluation.py          # Test set accuracy, timing, and confusion matrix computation
│
├── sample_clothing/           # Pre-saved sample images for all 10 clothing categories
│   ├── 0_t-shirt_top.png
│   ├── ...
│   └── 9_ankle_boot.png
│
└── data/
    └── fashion-mnist/         # Local dataset cache
```

---

## 🚀 Setup & Execution Guide

### Prerequisites
- Python 3.10+ (Python 3.11 recommended)
- Visual Studio Code or any terminal

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Run the Web Application
```powershell
streamlit run app.py
```
Open `http://localhost:8501` in your web browser.

---

## 🖥️ Web Application Features

1. **📖 Overview & Dataset**: Inspect subsampled $14 \times 14$ dataset images and review the mathematical formulations of Euclidean and Chi-Square distances.
2. **📊 Compare Performance**: Review live accuracy metrics, execution times, confusion matrices, and individual test predictions side-by-side.
3. **🎯 Test Clothing Prediction**: Upload custom clothing images or select verified samples across all 10 categories to inspect predictions, closest-neighbor visual matches, and full distance breakdowns.
