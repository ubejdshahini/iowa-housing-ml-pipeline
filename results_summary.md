# Iowa Housing ML Pipeline — Final Results

This document summarizes the final model results, feature justifications, performance metrics, and deliverables for **Person 2 (Features & Model)**.

---

## 1. Selected Features & Justification

The final pipeline uses 9 features (7 raw + 2 engineered). Below are their Pearson correlation coefficients with the target variable (`SalePrice`) and the rationale for their inclusion:

| Feature Name | Type | Correlation with SalePrice | Presentation Justification |
| :--- | :---: | :---: | :--- |
| **TotalFlrSF** | Engineered | **`+0.7169`** | Combined above-ground floor area is the single strongest predictor of price in the dataset. |
| **1stFlrSF** | Raw | `+0.6059` | Primary driver of ground-level livable area. |
| **FullBath** | Raw | `+0.5607` | Key buyer priority and strong proxy for overall house quality. |
| **TotRmsAbvGrd** | Raw | `+0.5337` | Captures overall capacity and layout beyond just bedrooms. |
| **HouseAge** | Engineered | **`-0.5234`** | Strong negative correlation; older homes depreciate due to wear and tear. |
| **YearBuilt** | Raw | `+0.5229` | Newer construction adheres to modern building standards. |
| **2ndFlrSF** | Raw | `+0.3193` | Adds valuable upper-story living space. |
| **LotArea** | Raw | `+0.2638` | Larger land parcels command a premium. |
| **BedroomAbvGr** | Raw | `+0.1682` | Basic filtering criteria used by home buyers. |

---

## 2. Train / Test Split
- **Split Ratio:** 80% Training / 20% Testing (`random_state=1`)
- **Training Set Shape:** `(1168, 9)`
- **Test Set Shape:** `(292, 9)`

---

## 3. Model Performance & Evaluation (Test Set — 292 rows)

| Model Version | Features | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | R² Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Median Price)** | — | \$56,621 | \$85,142 | `-0.0164` | Benchmark |
| **Random Forest v1** | 7 | \$22,394 | \$33,526 | `0.8424` | Superseded |
| **Random Forest v2** | **9** | **\$22,010** | **\$32,947** | **`0.8478`** | **Final Selected Model** |

### Key Performance Highlights:
- **Winning Model:** Random Forest Regressor (`n_estimators=300, min_samples_leaf=2, random_state=1, n_jobs=-1`).
- **MAE Error Reduction vs. Baseline:** **\$34,611** (**61% error reduction**).
- **MAE Improvement vs. v1:** **\$384** (1.72% error reduction).
- **Variance Explained ($R^2$):** **84.78%** of sale price variance on unseen test data.

---

## 4. Artifact Verification
The exported artifacts were reloaded and verified on the sample test case:
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

## 5. Final Deliverables (Repo Root)
1. **`model.ipynb`** — Complete Jupyter notebook with all executed cells.
2. **`iowa_model.pkl`** — Final serialized Random Forest v2 model (~13.6 MB).
3. **`iowa_features.pkl`** — Final serialized list of 9 features in order (< 1 KB).
4. **`sample_houses.csv`** — 5 realistic test houses matching the 9 feature columns (< 1 KB).
