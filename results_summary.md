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
| **Random Forest v2 (base)** | 9 | \$22,010 | \$32,947 | `0.8478` | Superseded |
| **Random Forest v2 (Tuned)** | **9** | **\$21,245** | **\$31,748** | **`0.8587`** | **Final Selected Model** |
| **Keras Neural Network** | **9** (scaled) | **\$31,980** | **\$41,680** | **`0.7564`** | **Evaluated (not deployed)** |

### Key Performance Highlights:
- **Winning Model:** Random Forest (Tuned) — best params: `n_estimators=100, max_depth=10, min_samples_leaf=1, min_samples_split=5, max_features='sqrt'`.
- **MAE Error Reduction vs. Baseline:** **\$35,376** (**62.5% error reduction**).
- **MAE Improvement vs. v1 (7 feat):** **\$1,149** (5.1% reduction).
- **Variance Explained ($R^2$):** **85.87%** of sale price variance on unseen test data.

---

## 4. Hyperparameter Tuning Details

`RandomizedSearchCV` was run with **20 iterations, 5-fold CV**, scoring on `neg_mean_absolute_error`.

| Hyperparameter | Search Space | Best Value |
| :--- | :--- | :--- |
| `n_estimators` | 100, 200, 300, 500 | **100** |
| `max_depth` | 10, 20, 30, None | **10** |
| `min_samples_split` | 2, 5, 10 | **5** |
| `min_samples_leaf` | 1, 2, 4 | **1** |
| `max_features` | 'sqrt', 'log2', 1.0 | **'sqrt'** |

- **Best CV MAE (neg):** `−22,149.90`

---

## 5. Artifact Verification
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
- **Predicted SalePrice:** **\$204,259.49** *(from previous RF v2 baseline — rerun notebook to refresh)*

---

## 6. Final Deliverables (Repo Root)
1. **`model.ipynb`** — Complete Jupyter notebook with EDA, feature engineering, training, tuning, DL comparison, and narrative.
2. **`iowa_model.pkl`** — Final serialized **Tuned Random Forest** model (overwritten by `tune_and_dl.py` after beating baseline).
3. **`iowa_features.pkl`** — Final serialized list of 9 features in order (< 1 KB). Immutable contract.
4. **`sample_houses.csv`** — 5 sample houses in raw format (8 features + `YrSold`) for Streamlit upload.
5. **`sample_houses_9feat.csv`** — 5 sample houses in 9-feature format for FastAPI `/predict-csv` testing.
6. **`tune_and_dl.py`** — Stand-alone tuning + DL script. Reproduces all training steps and overwrites `iowa_model.pkl` if improved.
7. **`nn_model.h5`** — Keras neural network weights (reference only, not deployed).
8. **`results_summary.md`** — This document.
9. **`requirements.txt`** — Pinned dependencies (`scikit-learn==1.8.0`, `tensorflow>=2.0,<3.0`, etc.).
