# 🌲 CoverType Forest Classifier

Multiclass classification project that predicts forest cover type from cartographic variables using the [UCI CoverType dataset](https://archive.ics.uci.edu/dataset/31/covertype).

---

## Dataset

581,012 samples · 54 features · 7 classes — collected from the Roosevelt National Forest, Colorado.

| Class | Cover Type |
|---|---|
| 1 | Spruce/Fir |
| 2 | Lodgepole Pine |
| 3 | Ponderosa Pine |
| 4 | Cottonwood/Willow |
| 5 | Aspen |
| 6 | Douglas-fir |
| 7 | Krummholz |

> ⚠️ Classes 1 & 2 account for ~84% of the data — significant class imbalance.

---

## Workflow

1. **Data Cleaning** — drop unnamed columns, check for nulls and duplicates
2. **EDA** — distributions, outlier detection (IQR), skewness, correlation heatmaps, scatter plots
3. **Preprocessing** — `RobustScaler` on continuous features, stratified 80/20 train/test split
4. **Modeling** — Random Forest and XGBoost with `RandomizedSearchCV` hyperparameter tuning
5. **Evaluation** — Accuracy, Macro F1, Confusion Matrix, Classification Report

---

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Random Forest | 93.89% | 90.42% |
| XGBoost | 88.01% | 87.82% |


---

## Requirements

```bash
pip install pandas numpy matplotlib seaborn plotly scipy scikit-learn xgboost tensorflow
```

---

## Usage

1. Place `covertype.csv` in the same directory as the notebook
2. Open `elevvo1.ipynb` and run all cells top to bottom

> ⚠️ XGBoost uses `device="cuda"` by default. Change to `device="cpu"` if no GPU is available.
