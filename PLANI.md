# PLANI (i përditësuar) — Iowa Housing ML Pipeline

**Projekti: Feature Engineering — FastAPI + Streamlit + Databricks**

Ky është plani origjinal i përditësuar sipas punës së kryer deri më tani (gusht 2026). Pjesa 1 (Notebook: analiza + feature engineering + modeli) është **e përfunduar**. Pjesa 2 (Deployment) dhe Pjesa 3 (Prezantimi) janë hapat në vazhdim.

---

## 0. Statusi aktual — çfarë është bërë

| Faza | Përgjegjës | Statusi |
| :--- | :--- | :---: |
| EDA (analiza e të dhënave) | Person 1 | ✅ E kryer |
| Features + Modeli (v1, 7 features) | Person 2 | ✅ E kryer |
| Feature engineering v2 (9 features) | Person 2 | ✅ E kryer |
| Krahasimi v1 vs v2 (grafik në notebook) | Person 2 | ✅ E kryer |
| Backend (FastAPI + Databricks) | Person 3 | ⏳ in progress |
| Frontend (Streamlit) + Prezantimi | Person 4 | ⏳ Në vazhdim |

**Deliverables ekzistuese në repo:** `eda.py`, `model.ipynb`, `iowa_model.pkl`, `iowa_features.pkl`, `sample_houses.csv`, `results_summary.md`.

---

## 1. KONTRATA E RE E FEATURES (kritike për të gjithë)

**Rregulli i artë ka ndryshuar:** modeli final nuk përdor më 7, por **9 features** (7 origjinale + 2 të inxhinieruara). I njëjti rend duhet të përputhet në notebook, FastAPI, Streamlit dhe Databricks:

```
FEATURES = [
    'LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF',
    'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd',
    'HouseAge',    # = YrSold - YearBuilt   (e re, v2)
    'TotalFlrSF',  # = 1stFlrSF + 2ndFlrSF  (e re, v2)
]
TARGET = 'SalePrice'
```

Kushdo që merr modelin duhet ta trajtojë këtë listë 9-she si "kontratë" fikse.

---

## PJESA 1 — Notebook (✅ E PËRFUNDUAR)

Përgjegjës: Person 1 + Person 2. Rezultatet reale të arritura:

- **Të dhënat (Person 1):** `data/train.csv` (1460 rreshta, 81 kolona, target `SalePrice`). EDA e plotë në `eda.py` — shape, tipet, statistikat, vlerat që mungojnë, korrelacionet, histograma e SalePrice, outliers.
- **Features (Person 2):** 7 features fillestare të përzgjedhura dhe të arsyetuara me korrelacion; vlerat që mungojnë të mbushura me medianë.
- **Modelet:** Baseline (median), Decision Tree, Random Forest — të krahasuara me MAE/RMSE/R² në 80/20 split (`random_state=1`).
- **Feature engineering v2:** u shtuan `HouseAge` dhe `TotalFlrSF`; modeli u ritrajnua me 9 features.

**Rezultatet finale (test set, 292 rreshta):**

| Modeli | Features | MAE | RMSE | R² |
| :--- | :---: | :---: | :---: | :---: |
| Baseline | — | $56,621 | $85,142 | -0.0164 |
| Random Forest v1 | 7 | $22,394 | $33,526 | 0.8424 |
| **Random Forest v2 (final)** | **9** | **$22,010** | **$32,947** | **0.8478** |

Feature engineering-u uli MAE-në me ~$384 (1.7%) dhe ngriti R² nga 0.8424 në 0.8478 — përmirësim i vogël por real. Grafiku krahasues v1 vs v2 është brenda `model.ipynb`.

**Deliverables të dorëzuara te Person 3:** `iowa_model.pkl`, `iowa_features.pkl` (9 features), `sample_houses.csv` (9 kolona).

---

## PJESA 2 — Deployment (⏳ IN Progress)

Përgjegjës kryesor: **Person 3 + Person 4**. Merrni setup-in sample (`github.com/xoniks/databricks-fastapi-streamlit`) dhe zëvendësoni modelin me tonin. Hapat mbeten si në planin origjinal, me **këto ndryshime të detyrueshme për shkak të kontratës 9-feature:**

### Hapi A — Zëvendëso modelin 
Kopjoni `iowa_model.pkl` dhe `iowa_features.pkl` (9 features) në rrënjë të projektit.

### Hapi B — Rregullo klasën Pydantic në `main.py` (KUJDES: gabim i njohur)
Emrat e atributeve në Python **nuk mund të fillojnë me shifër**, prandaj `1stFlrSF`/`2ndFlrSF` duhen ekspozuar si aliase, dhe duhen shtuar 2 features e reja:

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
    HouseAge: int      # e re (v2)
    TotalFlrSF: int    # e re (v2)

    class Config:
        populate_by_name = True
```

Para `model.predict(...)`, ktheni të dhënat në DataFrame me rendin e saktë të `iowa_features.pkl`:
```python
d = row.model_dump(by_alias=True)   # çelësat bëhen '1stFlrSF','2ndFlrSF', ...
df = pd.DataFrame([d])[features]     # rendi përputhet me .pkl
```

### Hapi C — Pin sklearn version
Modeli u ruajt me scikit-learn **1.8.0**. Vendoseni të njëjtin version në `requirements.txt` që ngarkimi i `.pkl` të mos japë version-mismatch warning.

### Hapi D — Tabela në Databricks (2 kolona shtesë)
Shtoni `house_age` dhe `total_flr_sf` në skemën e tabelës `iowa_preditions` (kujdes edhe me gabimin e njohur të shtypit 'preditions'):
```sql
CREATE TABLE IF NOT EXISTS workspace.default.iowa_preditions (
  lot_area DOUBLE, year_built DOUBLE,
  first_floor_sf DOUBLE, second_floor_sf DOUBLE,
  full_bath INT, bedroom_above_gr INT,
  total_rooms_above_grd INT,
  house_age INT, total_flr_sf DOUBLE,   -- kolonat e reja (v2)
  predicted_price DOUBLE
);
```

### Hapi E — FastAPI: nisje dhe test
`uvicorn main:app --reload` → hap `http://127.0.0.1:8000/docs` → provo `/predict-csv` me `sample_houses.csv` (9 kolona) dhe API key te 'Authorize'.

### Hapi F — Streamlit (Person 4)
Në `app/streamlit_app.py`: shto 2 fusha të reja në formular, ose llogariti automatikisht `HouseAge = YrSold - YearBuilt` dhe `TotalFlrSF = 1stFlrSF + 2ndFlrSF`. Dërgo payload-in me 9 features. Konfiguro `app/.streamlit/secrets.toml` me të njëjtin API key si `.env`.

### Hapi G — Test i plotë end-to-end
`streamlit run app/streamlit_app.py` → ngarko `sample_houses.csv` → Predict → Save to Databricks → verifiko me SELECT në tabelë.

---

## PJESA 3 — Prezantimi (⏳ NË VAZHDIM)

Përgjegjës: **Person 4**, me ndihmën e të gjithëve. Struktura e sugjeruar e slajdeve (e përditësuar):

- Hyrje — problemi dhe arkitektura (diagram).
- Të dhënat — nga vijnë, EDA (Person 1).
- Feature engineering — 7 features origjinale + PSE, dhe **v2: HouseAge, TotalFlrSF** + arsyet.
- Modeli — Baseline vs Decision Tree vs Random Forest; **grafiku v1 vs v2** (tashmë në notebook).
- Demo live — Streamlit → Predict → Save në Databricks.
- Përfundim — çka mësuam; përmirësimi i vogël por real nga feature engineering-u.

---

## Lista e kontrollit final (e përditësuar)

- [x] `train.csv` në `data/`.
- [x] Notebook-u ekzekutohet nga fillimi në fund pa gabime.
- [x] `iowa_model.pkl` dhe `iowa_features.pkl` (9 features) të krijuara dhe testuara.
- [x] `sample_houses.csv` me 9 kolonat e sakta.
- [x] Grafiku i krahasimit v1 vs v2 në notebook.
- [ ] FastAPI: klasa Pydantic me aliase + 9 features; `/predict-csv` kthen çmime.
- [ ] `requirements.txt` me sklearn version të pinuar.
- [ ] Tabela `iowa_preditions` me 2 kolonat e reja në Databricks.
- [ ] `/save-predictions` ruan rreshta që duken në Databricks.
- [ ] Streamlit lidhet me API-n, dërgon 9 features, bën parashikime + ruajtje.
- [ ] Slajdet gati dhe prova gjenerale e bërë.

---

## Gabimet më të shpeshta (të përditësuara)

- **Emrat e kolonave nuk përputhen** — CSV/JSON duhet të kenë saktësisht 9 features si në `iowa_features.pkl`.
- **`1stFlrSF`/`2ndFlrSF` si emra atributesh Python** — nuk lejohen; përdorni aliase (`FirstFlrSF`/`SecondFlrSF`).
- **Harresa e 2 features të reja** — nëse FastAPI/Streamlit/Databricks nuk shtojnë `HouseAge` dhe `TotalFlrSF`, deploy-i prishet.
- **Version mismatch i sklearn** — pinoni 1.8.0.
- **401 Unauthorized** — API key i Streamlit ≠ ai i `.env`.
- **Gabim emri tabele në Databricks** — kontrolloni 'iowa_preditions' (gabim shtypi i qëllimshëm në kodin sample).
