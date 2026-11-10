# Nearest Neighbor vs Histogram Classifier

## Project: Low-Resolution Clothing Image Classification (Fashion-MNIST)
Comparing **1-Nearest Neighbor** vs. **Histogram Classifier** for low-resolution ($14 \times 14$ pixels) image classification on the **Fashion-MNIST** benchmark dataset.

---

## 💡 Troubleshooting: "It is not identifying correctly — what to do?"

If your uploaded clothing photo was misclassified, here is the exact reason and how to fix it:

### The Problem:
1. **The Inversion Problem**:
   - In Fashion-MNIST, the **background is pitch black (pixel value 0.0)** and the **clothing item is light/white (values up to 1.0)**.
   - Real-world photos almost always have a **white or light-colored background**!
   - Because 1-Nearest Neighbor calculates Euclidean distance across all 196 pixels, a white background adds an overwhelming distance penalty ($\approx 140$ mismatched pixels), forcing the classifier to make wrong predictions.
2. **The Histogram Problem**:
   - A white background dumps almost 80–90% of all pixels into the brightest histogram bin ($1.0$), which never matches any Fashion-MNIST class template.
3. **The 14 × 14 Low-Resolution Constraint**:
   - Images are downsampled to only $14 \times 14$ pixels (196 values). If an item is off-center or has large white borders, the pixel alignment is lost.

### The Solution (Already Implemented in the App):
In **Tab 3 ("Test Clothing Prediction")**:
1. **Background Color Handling**:
   - Set to **"Auto (Detect light background and invert)"** or **"Force Invert (White BG -> Black)"**.
   - This automatically converts white backgrounds to black so they match Fashion-MNIST!
2. **Auto-Center & Crop**:
   - Keep **"Auto-center & crop clothing item"** enabled. This automatically detects the clothing silhouette and centers it in the frame.
3. **Shadow / Noise Filter**:
   - Adjust the noise slider ($0.02 - 0.08$) to clear faint camera shadows around the item.
4. **Instant Verification via Sample Clothing**:
   - Switch the radio button to **"🎯 Choose a pre-verified sample clothing item"**.
   - You can test any of the 10 classes (`T-shirt`, `Trouser`, `Pullover`, `Dress`, `Coat`, `Sandal`, `Sneaker`, `Bag`, etc.) and verify that the 1-NN model identifies them with high accuracy!

---

## 🧠 Algorithms Explained

1. **1-Nearest Neighbor (1-NN)**:
   - Flattens each $14 \times 14$ image into a 196-value vector.
   - Computes Euclidean distance $d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum (x_i - y_i)^2}$ to all training images and assigns the label of the closest sample.
   - **Performance**: High accuracy (~**78% – 82%** on test set) because it preserves spatial pixel patterns.
2. **Histogram Classifier**:
   - Extracts a normalized 16-bin grayscale intensity histogram from the image.
   - Compares with pre-calculated class average templates using **Chi-Square Distance**:
     $$\chi^2(P, Q) = \frac{1}{2} \sum_{i=1}^{16} \frac{(P_i - Q_i)^2}{P_i + Q_i + \epsilon}$$
   - **Performance**: Low accuracy (~**25% – 30%**) because histograms completely discard spatial layout (e.g. a bag and a folded shirt with the same brightness distribution yield the same histogram).

---

## 🛠️ Step-by-Step Execution Guide (Windows VS Code)

### Step 1: Open the Project in VS Code
1. Open VS Code.
2. Select **File > Open Folder...** and choose `c:\Users\shank\OneDrive\Desktop\cv-assign2`.

### Step 2: Open Terminal in VS Code
- Press ``Ctrl + ` ``.

### Step 3: Run the Streamlit Application
```powershell
streamlit run app.py
```
*(Or if using a virtual environment: `.\venv\Scripts\python.exe -m streamlit run app.py`)*

The browser will open to `http://localhost:8501`.

---

## 📋 What the Web App Shows

1. **📖 Tab 1: Overview & Dataset**:
   - Visual display of 10 clothing categories at $14 \times 14$ resolution.
   - Mathematical descriptions of Euclidean Distance and Chi-Square Distance.
2. **📊 Tab 2: Compare Performance**:
   - Accuracy and evaluation time comparison cards.
   - Confusion matrices for both 1-NN and Histogram methods.
   - Side-by-side test sample predictions with ground truth.
3. **🎯 Tab 3: Test Clothing Prediction**:
   - Upload any custom photo or pick from 10 verified category samples.
   - Visual inspection of the centered, inverted, and downsampled images.
   - Displays the **exact closest training sample matched by 1-NN** and distance score!
   - Full Chi-Square distance breakdown across all 10 categories.
