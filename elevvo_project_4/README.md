# 🚦 Traffic Sign Recognition — GTSRB

A deep learning project that classifies **43 types of German traffic signs** from images using a custom CNN and MobileNetV2, trained on the GTSRB benchmark dataset.

---

## 📌 Project Overview

| | |
|---|---|
| **Dataset** | [GTSRB — German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) |
| **Task** | Multi-class image classification (43 classes) |
| **Models** | Custom CNN · MobileNetV2 |
| **Platform** | Kaggle Notebooks (GPU T4) |
| **Best Accuracy** | 97.51% (Custom CNN) |

---

## 📁 Dataset

The GTSRB dataset contains over **50,000 real-world traffic sign images** across 43 classes with significant variation in lighting, angle, blur, and size.

```
GTSRB/
├── Train/
│   ├── 0/        ← Speed limit 20
│   ├── 1/        ← Speed limit 30
│   └── ...       ← up to class 42
├── Test/
└── Test.csv
```

---

## ⚙️ Pipeline

### 1. Preprocessing
- Images resized to `64×64` pixels
- Pixel values normalized from `[0, 255]` → `[0, 1]`
- Labels one-hot encoded into vectors of size 43
- 80/20 train/validation split with stratification

### 2. Data Augmentation
Applied to balance underrepresented classes up to 1000 images each:
- Random rotation (±15°)
- Width & height shifts (±10%)
- Zoom (±10%)
- Brightness variation (0.8–1.2×)
- No horizontal flip — traffic signs are not symmetric

Class weights also applied during training to further handle imbalance.

### 3. Custom CNN Architecture
```
Input (64×64×3)
  ↓
Conv2D(32) + BatchNorm + Conv2D(32) + MaxPool + Dropout(0.25)
  ↓
Conv2D(64) + BatchNorm + Conv2D(64) + MaxPool + Dropout(0.25)
  ↓
Conv2D(128) + BatchNorm + MaxPool + Dropout(0.25)
  ↓
Flatten → Dense(256) + Dropout(0.5) → Dense(43, softmax)
```

### 4. MobileNetV2 (Transfer Learning)
- Input resized to `96×96` for compatibility
- Pre-trained on ImageNet with top layers removed
- Custom classification head added
- Two-phase training: freeze base → fine-tune top 30 layers

### 5. Training Setup
| Setting | Value |
|---|---|
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Batch Size | 64 |
| Max Epochs | 30 |
| Early Stopping | patience=15 |
| LR Reduction | factor=0.5, patience=4 |

---

## 📊 Results

| Model | Test Accuracy |
|---|---|
| Custom CNN | **97.51%** |
| MobileNetV2 | **97.21%** |

The custom CNN outperformed MobileNetV2 in this case because it was designed specifically for `64×64` inputs, while MobileNetV2 was originally built for larger images (`224×224`).

---

## 🗂️ Notebook Structure

| Section | Description |
|---|---|
| Import Libraries | All dependencies |
| Settings | Paths, image size, batch size, epochs |
| Load & Preprocess | Load images from class folders, normalize, split |
| Sanity Check | Verify shapes and pixel values |
| Explore the Data | Class distribution bar chart, sample images |
| Data Augmentation | Balance classes, visualize augmented images |
| Build Custom CNN | Define and compile CNN architecture |
| Train Custom CNN | Fit with callbacks and class weights |
| Build & Train MobileNetV2 | Transfer learning with two-phase training |
| Plot Training Curves | Accuracy and loss curves for both models |
| Load Test Data | Load and preprocess test images from CSV |
| Evaluate Both Models | Accuracy, loss, comparison bar chart |
| Confusion Matrix | Heatmap + classification report per class |
| Visualize Predictions | Sample predictions with green/red labels |
| Save Models | Export to `.keras` format |

---

## 🛠️ Technologies

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange)
![Keras](https://img.shields.io/badge/Keras-3.x-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-yellow)

| Library | Usage |
|---|---|
| TensorFlow / Keras | Model building and training |
| OpenCV | Image loading and preprocessing |
| NumPy | Array operations |
| Pandas | CSV handling and EDA |
| Scikit-learn | Train/val split, metrics, class weights |
| Matplotlib / Seaborn | Plots, confusion matrix heatmap |

---

## 🚀 How to Run

### On Kaggle
1. Open a new Kaggle notebook
2. Click **+ Add Data** → search `GTSRB German Traffic Sign` → add it
3. Go to **Settings → Accelerator → GPU T4**
4. Upload and run the notebook

### On Google Colab
1. Upload your GTSRB zip to Google Drive
2. Mount Drive and extract the zip
3. Go to **Runtime → Change runtime type → GPU**
4. Run all cells

### Locally
```bash
pip install tensorflow opencv-python scikit-learn matplotlib seaborn pandas numpy
```
Update the paths in the Settings cell to point to your local dataset folder, then run all cells.

---

## 🏷️ Classes (43 Traffic Sign Categories)

```
0  Speed 20        1  Speed 30        2  Speed 50        3  Speed 60
4  Speed 70        5  Speed 80        6  End 80          7  Speed 100
8  Speed 120       9  No Passing     10  No Passing>3.5t 11  Right of Way
12 Priority Road  13  Yield          14  Stop            15  No Vehicles
16 No Trucks      17  No Entry       18  Danger          19  Left Curve
20 Right Curve    21  Double Curve   22  Bumpy Road      23  Slippery Road
24 Road Narrows   25  Road Work      26  Traffic Signals 27  Pedestrians
28 Children       29  Cyclists       30  Ice/Snow        31  Wild Animals
32 End Restrict.  33  Turn Right     34  Turn Left       35  Go Ahead
36 Go Right/Ahead 37  Go Left/Ahead  38  Keep Right      39  Keep Left
40 Roundabout     41  End No Passing 42  End No Pass>3.5t
```
