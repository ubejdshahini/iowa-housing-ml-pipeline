# Iowa Housing ML Pipeline — Features & Model Results Summary (v1 vs v2)

This document summarizes the model development process, feature justifications, evaluation metrics, and final deliverables for both **v1 (7 features)** and **v2 (9 features)**.

---

## 1. Selected Features & Justification

The final pipeline uses exactly 9 features (7 original + 2 engineered). Below are their Pearson correlation coefficients with the target variable (`SalePrice`) and the rationale for their inclusion:

| Feature | Correlation with SalePrice | Presentation Justification |
| :--- | :---: | :--- |
| **TotalFlrSF** (v2) | **`0.7169`** | Combined above-ground floor area is a significantly stronger size proxy than individual floors. |
| **1stFlrSF** | `0.6059` | First-floor square footage is the primary driver of interior space. |
| **FullBath** | `0.5607` | Full bathrooms are a key buyer priority — strong proxy for quality. |
| **TotRmsAbvGrd** | `0.5337` | Total rooms capture overall house size beyond just bedrooms. |
| **HouseAge** (v2) | **`-0.5234`** | Strong negative correlation indicating older houses depreciate in value. |
| **YearBuilt** | `0.5229` | Newer homes are in better condition and built to modern standards. |
| **2ndFlrSF** | `0.3193` | Additional floor area directly increases livable space and price. |
| **LotArea** | `0.2638` | Larger lots command a premium — more land = more value. |
| **BedroomAbvGr** | `0.1682` | Bedroom count is one of the first filters buyers use when searching. |

---

## 2. Train / Test Split
- **Split Ratio:** 80% Training / 20% Testing (`random_state=1`)
- **Training Set Shape (v2):** `(1168, 9)`
- **Test Set Shape (v2):** `(292, 9)`

---

## 3. Evaluation & Model Comparison (Test Set)

Metrics computed on the test set (292 rows):

| Model Version | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | R² Score |
| :--- | :---: | :---: | :---: |
| **Baseline (Median Price)** | \$56,621 | \$85,142 | `-0.0164` |
| **Random Forest v1 (7 features)** | \$22,394 | \$33,526 | `0.8424` |
| **Random Forest v2 (9 features — Winner)** | **\$22,010** | **\$32,947** | **`0.8478`** |

### Winner Selection Decision:
The **Random Forest v2** model is selected as the final model because it achieves the lowest MAE (**\$22,010**).
- **MAE Reduction:** \$384 (**1.72%** error reduction vs. v1).
- **Total MAE Improvement vs. Baseline:** **\$34,611 (61% error reduction)**.
- **R² Score:** Explains **84.78%** of the variance in sales prices on unseen data.

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
    'TotRmsAbvGrd': 8,
    'HouseAge': 5,
    'TotalFlrSF': 1710
}
```
- **Loaded Model Type:** `RandomForestRegressor`
- **Predicted SalePrice:** **\$204,259.49**

---

## 5. Deliverables Generated (saved to repo root)
1. **`iowa_model.pkl`** (~13.6 MB) — Python serialized Random Forest v2 model.
2. **`iowa_features.pkl`** (< 1 KB) — Python serialized list of the 9 features in the correct order.
3. **`sample_houses.csv`** (< 1 KB) — 5 realistic houses with all 9 features for Streamlit UI testing.

---

## 6. Downstream Deployment Impact Warning Checklist

⚠️ **To prevent runtime breakages, the following updates must be made by Person 3 and Person 4:**

- [ ] **FastAPI Backend (`main.py`) — Person 3**
  - Add `'HouseAge'` and `'TotalFlrSF'` to the model feature list.
  - Update `PredictionRow`/`HouseData` Pydantic class to include `HouseAge: int` and `TotalFlrSF: int`.
- [ ] **Streamlit Frontend App — Person 3**
  - Add user input widgets for the new features.
  - Alternatively, compute them dynamically in the UI code:
    - `HouseAge = YrSold - YearBuilt`
    - `TotalFlrSF = 1stFlrSF + 2ndFlrSF`
  - Append `'HouseAge'` and `'TotalFlrSF'` to the payload JSON sent to FastAPI.
- [ ] **Databricks Pipeline / SQL Schema — Person 4**
  - Add `HouseAge` and `TotalFlrSF` columns to the prediction tracking schema/table.
