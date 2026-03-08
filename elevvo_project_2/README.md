# Loan Approval Prediction

A binary classification project that predicts whether a loan application will be **Approved** or **Rejected** based on applicant financial and personal profile data.

---

## Overview

| | |
|---|---|
| **Task** | Binary Classification |
| **Dataset** | `loan_approval_dataset.csv` |
| **Samples** | 4,269 |
| **Features** | 12 |
| **Target** | `loan_status` — Approved / Rejected |
| **Class Imbalance Handling** | SMOTE |
| **Models** | Logistic Regression · Random Forest |

---

## Dataset

| Feature | Type | Description |
|---|---|---|
| `no_of_dependents` | Numeric | Number of financially dependent individuals |
| `education` | Categorical | Graduate / Not Graduate |
| `self_employed` | Categorical | Yes / No |
| `income_annum` | Numeric | Annual income of the applicant |
| `loan_amount` | Numeric | Total loan amount requested |
| `loan_term` | Numeric | Loan repayment duration |
| `cibil_score` | Numeric | Credit score (300–900) — strongest predictor |
| `residential_assets_value` | Numeric | Market value of residential property |
| `commercial_assets_value` | Numeric | Market value of commercial property |
| `luxury_assets_value` | Numeric | Value of luxury assets |
| `bank_asset_value` | Numeric | Bank balance and fixed deposits |
| `loan_status` | Target | Approved (1) / Rejected (0) |

---

## Project Pipeline

```
Data Cleaning → EDA → Outlier Detection → Encoding → Correlation Analysis
→ Scaling → Feature Selection → Modeling → SMOTE → Modeling → Evaluation
```

### 1. Data Cleaning
- Drop `loan_id` and strip whitespace from column names
- Verify no missing values or duplicate rows

### 2. Exploratory Data Analysis
- Target class distribution
- Feature distributions with KDE
- Annual income vs. loan amount scatter plot colored by loan status
- CIBIL score vs. loan status boxplot
- Loan status breakdown by education level

### 3. Outlier Detection
- Z-Score method (threshold = 3.0)
- IQR method (factor = 1.5)
- Boxplots showing mean, median, and outlier count per feature

### 4. Encoding & Correlation
- Label encode `loan_status`, `education`, and `self_employed`
- Pearson heatmap — linear correlations
- Kendall heatmap — non-linear relationships

### 5. Preprocessing
- Scale all numeric features with `RobustScaler`
- Stratified 80/20 train/test split

### 6. Feature Selection
- Mutual Information scores ranked and visualized for all features

### 7. Modeling
Both models are trained and evaluated **twice** — on the original data and on SMOTE-resampled data.

| Model | Key Parameters |
|---|---|
| Logistic Regression | `penalty=l1`, `solver=saga`, `C=0.1`, `class_weight=balanced` |
| Random Forest | `n_estimators=250`, `class_weight=balanced` |

### 8. Evaluation
- Accuracy · ROC-AUC Score · ROC Curve
- Confusion Matrix · Classification Report

---

## Results

| Model | SMOTE | Accuracy | ROC-AUC |
|---|---|---|---|
| Logistic Regression | ✗ | 94.15% | 97.42% |
| Random Forest | ✗ | 98.01% | 99.89% |
| Logistic Regression | ✓ | 93.44% | 97.41% |
| Random Forest | ✓ | 98.01% | 99.87% |


---

## Requirements

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn imbalanced-learn
```

---

## Usage

1. Place `loan_approval_dataset.csv` in the same directory as the notebook
2. Open `elevvo2.ipynb` in Jupyter or Google Colab
3. Run all cells top to bottom
