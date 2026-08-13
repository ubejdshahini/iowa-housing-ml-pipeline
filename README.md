# Iowa Housing ML Pipeline

Feature engineering & ML pipeline for Iowa house price prediction — nga EDA te FastAPI + Streamlit + Databricks deployment.

Qëllimi: parashikojmë **çmimin e shtëpive** në Iowa me të dhëna reale nga Kaggle, dhe pastaj e vendosim modelin **online** në një aplikacion ku përdoruesi ngarkon një skedar dhe merr parashikimet.

---

## Arkitektura

```
Dataset (Kaggle)  ->  Notebook (analiza + model)  ->  iowa_model.pkl
                                                          |
                                                          v
     Streamlit (web)  <->  FastAPI (API)  ->  Databricks (baza e të dhënave)
```

- **Notebook** — analizojmë të dhënat, zgjedhim/krijojmë features, trajnojmë modelin, e ruajmë si `.pkl`.
- **Model (.pkl)** — 'truri' i trajnuar, i ngarkuar pa u ritrajnuar çdo herë.
- **FastAPI** — ngarkon modelin dhe kthen parashikime kur i dërgohet një kërkesë.
- **Streamlit** — faqja web ku përdoruesi ngarkon një CSV dhe sheh rezultatet.
- **Databricks** — ruan parashikimet në një tabelë.

---

## Statusi

| Faza | Përgjegjës | Statusi |
| :--- | :--- | :---: |
| EDA (analiza e të dhënave) | Person 1 | ✅ |
| Features + Modeli (v1, 7 features) | Person 2 | ✅ |
| Feature engineering v2 (9 features) | Person 2 | ✅ |
| Krahasimi v1 vs v2 (grafik në notebook) | Person 2 | ✅ |
| Backend (FastAPI + Databricks) | Person 3 | ✅ |
| Frontend (Streamlit) + Prezantimi | Person 4 | ⏳ |

---

## Struktura e repo-s

```
iowa-housing-ml-pipeline/
├── data/train.csv          # dataset-i (1460 rreshta, 81 kolona)
├── eda.py                  # analiza eksploruese (Person 1)
├── model.ipynb             # features + modeli + v2 + grafiku (Person 2)
├── iowa_model.pkl          # modeli final Random Forest v2
├── iowa_features.pkl       # lista e 9 features (rendi i saktë)
├── sample_houses.csv       # 5 shtëpi shembull (9 kolona)
├── main.py                 # FastAPI: predict-csv + ruajtja në Databricks
├── .env.example            # konfigurimi shembull pa sekrete
├── results_summary.md      # përmbledhja e rezultateve
├── requirements.txt
└── README.md
```

---

## Pergatitjet — çfarë duhet të instalojë secili

- **Python 3.10+**, **VS Code / Jupyter**, **Git**, llogari **Kaggle**, qasje në **Databricks**.
- Secili instalon vetëm çka i duhet për pjesën e vet.

---

## Ndarja e punës (4 persona)

- **Person 1 — Të dhënat & EDA:** shkarkon datasetin, e kupton, kontrollon vlerat që mungojnë, korrelacionet.
- **Person 2 — Features & Modeli:** përzgjedh/krijon features, trajnon e krahason modelet, eksporton `.pkl`.
- **Person 3 — Backend (FastAPI + Databricks):** fut modelin, konfiguron `.env`, nis API-n dhe tabelën.
- **Person 4 — Frontend (Streamlit) & Prezantimi:** nis faqen web, e lidh me API-n, verifikon Databricks, përgatit slajdet.

---

## KONTRATA E FEATURES (9 features — kritike)

Modeli final përdor **9 features** (7 origjinale + 2 të inxhinieruara). I njëjti rend duhet të përputhet në notebook, FastAPI, Streamlit dhe Databricks:

```python
FEATURES = [
    'LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF',
    'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd',
    'HouseAge',    # = YrSold - YearBuilt   (e re, v2)
    'TotalFlrSF',  # = 1stFlrSF + 2ndFlrSF  (e re, v2)
]
TARGET = 'SalePrice'
```

---

## Rezultatet finale (test set — 292 rreshta)

| Modeli | Features | MAE | RMSE | R² |
| :--- | :---: | :---: | :---: | :---: |
| Baseline (median) | — | $56,621 | $85,142 | -0.0164 |
| Random Forest v1 | 7 | $22,394 | $33,526 | 0.8424 |
| **Random Forest v2 (final)** | **9** | **$22,010** | **$32,947** | **0.8478** |

Feature engineering-u uli MAE-në me ~$384 (1.7%) dhe ngriti R² nga 0.8424 në 0.8478. Grafiku krahasues v1 vs v2 gjendet brenda `model.ipynb`.

---

# PJESA 1 — Notebook (✅ E PËRFUNDUAR)

Përgjegjës: Person 1 + Person 2.

1. **Shkarko datasetin** nga Kaggle ('Housing Prices Competition') → `data/train.csv`.
2. **Përgatit ambientin** — `pip install pandas numpy scikit-learn matplotlib joblib seaborn`.
3. **EDA** — `df.shape`, `df.info()`, `df.describe()`, histogrami i `SalePrice`.
4. **Kontrollo vlerat që mungojnë** — `df.isna().sum()`.
5. **Përzgjedh features** — 7 kolona numerike të lidhura fort me çmimin (të arsyetuara me korrelacion).
6. **Pastro / krijo features të reja (v2)** — `HouseAge`, `TotalFlrSF`; mbush vlerat që mungojnë me medianë.
7. **Ndaj train/test** — 80/20, `random_state=1`.
8. **Model baze (baseline)** — parashiko medianën për çdo shtëpi.
9. **Trajno modelet** — Decision Tree (`max_depth=8, min_samples_leaf=5`) dhe Random Forest (`n_estimators=300, min_samples_leaf=2, n_jobs=-1`).
10. **Vlerëso & krahaso** — MAE, RMSE, R² në test set.
11. **Zgjidh modelin final** — ai me MAE më të vogël (Random Forest v2 fitoi).
12. **Eksporto** — `joblib.dump` për `iowa_model.pkl` dhe `iowa_features.pkl`.
13. **Verifiko** — ringarko `.pkl`-të dhe bëj një parashikim prove.
14. **Krijo `sample_houses.csv`** — disa shtëpi shembull me emrat e kolonave saktësisht si 9 features.

**Bonus (Person 2):** krahasimi v1 vs v2 me grafik (MAE/RMSE/R²) brenda `model.ipynb`.

---

# PJESA 2 — Deployment (Backend ✅, Streamlit ⏳ — Person 3 + Person 4)

Merrni setup-in sample `github.com/xoniks/databricks-fastapi-streamlit` dhe zëvendësoni modelin me tonin.

1. **Merr kodin** — `git clone` të repo-s sample; brenda: `main.py`, `app/streamlit_app.py`, `requirements.txt`, `.pkl` shembull.
2. **Ambienti virtual** — `python -m venv venv`, aktivizo, `pip install -r requirements.txt`.
3. **Zëvendëso modelin** — kopjo `iowa_model.pkl` + `iowa_features.pkl` (9 features) në rrënjë.
4. **Rregullo klasën Pydantic në `main.py`** — emrat e atributeve nuk mund të fillojnë me shifër, prandaj:

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
   Para `model.predict(...)`: `df = pd.DataFrame([row.model_dump(by_alias=True)])[features]`.

5. **Pin sklearn** — modeli u ruajt me scikit-learn **1.8.0**; vendose të njëjtin version në `requirements.txt`.
6. **Krijo `.env`** — kopjo `.env.example` në `.env` dhe plotëso `API_KEY`, `DATABRICKS_SERVER_HOSTNAME`, `DATABRICKS_HTTP_PATH`, `DATABRICKS_TOKEN` dhe `DATABRICKS_TABLE` (kurrë mos e ngarko `.env` në GitHub).
7. **Nis FastAPI** — `py -m uvicorn main:app --host 127.0.0.1 --port 8001 --reload` → `http://127.0.0.1:8001/docs` → provo `/predict-csv` me `sample_houses.csv` (9 kolona). API-ja ofron `GET /` për health check, `POST /predict-csv` për parashikime dhe `POST /save-predictions` për ruajtje në Databricks; dy endpoint-et POST mbrohen me header-in `X-API-Key`.
8. **Tabela në Databricks (2 kolona shtesë):**

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
   (Kujdes: emri 'iowa_preditions' ka gabim shtypi të qëllimshëm në kodin sample.)
9. **Testo `/save-predictions`** — pastaj `SELECT` në tabelë për të parë që u ruajtën.
10. **Konfiguro Streamlit** — `app/.streamlit/secrets.toml` me `FASTAPI_URL` dhe `FASTAPI_API_KEY` (i njëjti me `.env`). Shto 2 fushat e reja në formular ose llogariti `HouseAge`/`TotalFlrSF` automatikisht.
11. **Test i plotë** — `streamlit run app/streamlit_app.py` → ngarko CSV → Predict → Save to Databricks → verifiko.

---

# PJESA 3 — Prezantimi (⏳ NË VAZHDIM — Person 4)

Struktura e sugjeruar e slajdeve:

1. Hyrje — problemi dhe arkitektura (diagram).
2. Të dhënat — nga vijnë, çka pamë në EDA (Person 1).
3. Feature engineering — 7 features origjinale + PSE, dhe **v2: HouseAge, TotalFlrSF** + arsyet.
4. Modeli — Baseline vs Decision Tree vs Random Forest; **grafiku v1 vs v2**.
5. Demo live — Streamlit → Predict → Save në Databricks.
6. Përfundim — çka mësuam; përmirësimi i vogël por real nga feature engineering-u.

Bëni një provë gjenerale të plotë; mbani gati edhe screenshots si rezervë.

---

## Lista e kontrollit final

- [x] `train.csv` në `data/`.
- [x] Notebook-u ekzekutohet nga fillimi në fund pa gabime.
- [x] `iowa_model.pkl` + `iowa_features.pkl` (9 features) të krijuara dhe testuara.
- [x] `sample_houses.csv` me 9 kolonat e sakta.
- [x] Grafiku i krahasimit v1 vs v2 në notebook.
- [x] FastAPI: klasa Pydantic me aliase + 9 features; `/predict-csv` kthen çmime.
- [x] `requirements.txt` me sklearn version të pinuar.
- [x] Tabela `iowa_preditions` me 2 kolonat e reja në Databricks.
- [x] `/save-predictions` ruan rreshta që duken në Databricks.
- [ ] Streamlit dërgon 9 features, bën parashikime + ruajtje.
- [ ] Slajdet gati dhe prova gjenerale e bërë.

---

## Gabimet më të shpeshta

- **Emrat e kolonave nuk përputhen** — CSV/JSON duhet të kenë saktësisht 9 features si në `iowa_features.pkl`.
- **`1stFlrSF`/`2ndFlrSF` si emra atributesh Python** — nuk lejohen; përdorni aliase (`FirstFlrSF`/`SecondFlrSF`).
- **Harresa e 2 features të reja** — nëse FastAPI/Streamlit/Databricks s'shtojnë `HouseAge` dhe `TotalFlrSF`, deploy-i prishet.
- **Version mismatch i sklearn** — pinoni 1.8.0.
- **401 Unauthorized** — API key i Streamlit ≠ ai i `.env`.
- **Gabim emri tabele** — kontrolloni 'iowa_preditions'.
