# Iowa Housing ML Pipeline — Results & Deployment Warnings

This document contains the complete model results, feature justifications, evaluation metrics, and critical downstream deployment warnings for **Person 2 (Features & Model)**.

---

## 📊 1. Model Performance Results

### Model Comparison Table (Test Set — 292 rows):

| Model Version | Features | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | R² Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Median Price)** | — | \$56,621 | \$85,142 | `-0.0164` | Reference |
| **Random Forest v1** | 7 | \$22,394 | \$33,526 | `0.8424` | Superseded |
| **Random Forest v2** | **9** | **\$22,010** | **\$32,947** | **`0.8478`** | **Selected Winner** |

### Summary Highlights:
- **Winning Model:** Random Forest Regressor (`n_estimators=300, min_samples_leaf=2, random_state=1, n_jobs=-1`).
- **MAE Error Reduction vs. Baseline:** **\$34,611** (61% error reduction).
- **MAE Improvement vs. v1:** **\$384** (1.72% error reduction).
- **Variance Explained ($R^2$):** **84.78%** of sale price variance on unseen data.

---

## 🔍 2. Feature Justification & Pearson Correlations

| Feature Name | Type | Correlation with SalePrice | Presentation Rationale |
| :--- | :---: | :---: | :--- |
| **TotalFlrSF** | Engineered (v2) | **`+0.7169`** | Combined above-ground area is the single strongest predictor of price in the dataset. |
| **1stFlrSF** | Raw | `+0.6059` | Primary driver of ground-level livable area. |
| **FullBath** | Raw | `+0.5607` | Key buyer priority and strong proxy for overall house quality. |
| **TotRmsAbvGrd** | Raw | `+0.5337` | Captures overall capacity and layout beyond just bedrooms. |
| **HouseAge** | Engineered (v2) | **`-0.5234`** | Strong negative correlation; older homes depreciate due to wear and tear. |
| **YearBuilt** | Raw | `+0.5229` | Newer construction adheres to modern building standards. |
| **2ndFlrSF** | Raw | `+0.3193` | Adds valuable upper-story living space. |
| **LotArea** | Raw | `+0.2638` | Larger land parcels command a premium. |
| **BedroomAbvGr** | Raw | `+0.1682` | Basic filtering criteria used by home buyers. |

---

## ⚠️ 3. Downstream Deployment Impact Warnings

Because the production model was upgraded to **v2 (9 features)**, Person 3 and Person 4 **MUST** update their respective components before deployment:

### Checklist for Person 3 (FastAPI & Streamlit App):
- [ ] **FastAPI (`main.py`):**
  - Update `FEATURES` array to include `'HouseAge'` and `'TotalFlrSF'`.
  - Update `PredictionRow` / `HouseData` Pydantic class:
    ```python
    from pydantic import BaseModel, Field

    class PredictionRow(BaseModel):
        LotArea: int
        YearBuilt: int
        FirstFlrSF: int  = Field(alias="1stFlrSF")
        SecondFlrSF: int = Field(alias="2ndFlrSF")
        FullBath: int
        BedroomAbvGr: int
        TotRmsAbvGrd: int
        HouseAge: int
        TotalFlrSF: int

        class Config:
            populate_by_name = True
    ```
    > *Note for Person 3:* Python attribute names cannot start with a digit, so `1stFlrSF`/`2ndFlrSF` are exposed as Pydantic aliases (`alias="1stFlrSF"` / `alias="2ndFlrSF"`); the model's actual feature column names stay `1stFlrSF`/`2ndFlrSF`. (This is reference guidance for Person 3's implementation, not written code).
- [ ] **Streamlit App:**
  - Add input widgets (or compute on the fly: `HouseAge = YrSold - YearBuilt`, `TotalFlrSF = 1stFlrSF + 2ndFlrSF`).
  - Pass the updated 9-feature payload JSON to the FastAPI endpoint.

### Checklist for Person 4 (Databricks & SQL Logging):
- [ ] **Databricks Pipeline:**
  - Add `HouseAge` and `TotalFlrSF` columns to the prediction tracking schema/table.
  - Update SQL logging queries to log all 9 feature inputs per request.

---

## 📁 4. Final Deliverables (Repo Root)
- **`model.ipynb`** — Complete Jupyter notebook (v1 + v2 executed cells).
- **`iowa_model.pkl`** — Final serialized Random Forest v2 model.
- **`iowa_features.pkl`** — Final serialized list of 9 features in order.
- **`sample_houses.csv`** — 5 test houses with 9 feature columns.
