# Iowa Housing ML Pipeline — Features & Model Results Summary

This document summarizes the model development process, feature justifications, evaluation metrics, and final deliverables for **Person 2 — Features & Model**.

---

## 1. Selected Features & Justification

The pipeline uses exactly 7 features. Below are their Pearson correlation coefficients with the target variable (`SalePrice`) and the rationale for their inclusion:

| Feature | Correlation with SalePrice | Presentation Justification |
| :--- | :---: | :--- |
| **1stFlrSF** | `0.6059` | First-floor square footage is the primary driver of interior space. |
| **FullBath** | `0.5607` | Full bathrooms are a key buyer priority — strong proxy for quality. |
| **TotRmsAbvGrd** | `0.5337` | Total rooms capture overall house size beyond just bedrooms. |
| **YearBuilt** | `0.5229` | Newer homes are in better condition and built to modern standards. |
| **2ndFlrSF** | `0.3193` | Additional floor area directly increases livable space and price. |
| **LotArea** | `0.2638` | Larger lots command a premium — more land = more value. |
| **BedroomAbvGr** | `0.1682` | Bedroom count is one of the first filters buyers use when searching. |

---

## 2. Train / Test Split
- **Split Ratio:** 80% Training / 20% Testing (`random_state=1`)
- **Training Set Shape:** `(1168, 7)`
- **Test Set Shape:** `(292, 7)`

---

## 3. Evaluation & Model Comparison (Test Set)

Metrics computed on the test set (292 rows):

| Model | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | R² Score |
| :--- | :---: | :---: | :---: |
| **Baseline (Median Price)** | \$56,621 | \$85,142 | `-0.0164` |
| **Decision Tree** | \$26,968 | \$40,787 | `0.7667` |
| **Random Forest (Winner)** | **\$22,394** | **\$33,526** | **`0.8424`** |

### Winner Selection:
The **Random Forest Regressor** (`n_estimators=300, min_samples_leaf=2, random_state=1, n_jobs=-1`) is selected because it achieves the **lowest MAE (\$22,394)**. 
- **MAE Reduction:** \$34,227 (**60% reduction in prediction error** vs. Baseline).
- **R² Score:** Explains **84.24%** of the variance in sales prices on unseen data.

---

## 4. Artifact Verification
The exported artifacts were reloaded and verified on the following test case:
```python
{
    'LotArea': 8450,
    'YearBuilt': 2003,
    '1stFlrSF': 856,
    '2ndFlrSF': 854,
    'FullBath': 2,
    'BedroomAbvGr': 3,
    'TotRmsAbvGrd': 8
}
```
- **Loaded Model Type:** `RandomForestRegressor`
- **Predicted SalePrice:** **\$206,060**

---

## 5. Deliverables Generated (saved to repo root)
1. **`iowa_model.pkl`** (~13.6 MB) — Python serialized Random Forest model.
2. **`iowa_features.pkl`** (< 1 KB) — Python serialized list of the 7 features in the correct order.
3. **`sample_houses.csv`** (< 1 KB) — 5 realistic houses representing starter, average, family, luxury, and bungalow profiles for Streamlit UI testing.
