"""
tune_and_dl.py
==============
Iowa Housing ML Pipeline — Hyperparameter Tuning & Deep Learning Comparison
----------------------------------------------------------------------------
Purpose:
    Stand-alone script that reproduces the model selection experiments
    performed in model.ipynb:
      1. Tunes a Random Forest with RandomizedSearchCV.
      2. Trains a Keras neural network as a comparison baseline.
      3. Overwrites iowa_model.pkl if the tuned RF outperforms the
         previously exported model (MAE benchmark: 21,830.58).

Usage:
    python tune_and_dl.py

Outputs:
    - iowa_model.pkl   — updated only if tuned RF beats the benchmark
    - nn_model.keras   — Keras model weights (reference only, not deployed)

Notes:
    - The 9-feature contract (iowa_features.pkl) is NEVER modified.
    - The neural network is comparison-only; the deployed model is always
      a scikit-learn RandomForestRegressor loaded via joblib.
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. Constants
# ──────────────────────────────────────────────
TRAIN_PATH = os.path.join("data", "train.csv")
MODEL_PATH = "iowa_model.pkl"
NN_PATH = "nn_model.keras"
FEATURES = [
    "LotArea",
    "YearBuilt",
    "1stFlrSF",
    "2ndFlrSF",
    "FullBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "HouseAge",    # engineered: YrSold - YearBuilt
    "TotalFlrSF",  # engineered: 1stFlrSF + 2ndFlrSF
]
TARGET = "SalePrice"
RANDOM_STATE = 1
EXISTING_MAE_BENCHMARK = 21_830.58  # RF v2 baseline MAE


# ──────────────────────────────────────────────
# 2. Data loading & feature engineering
# ──────────────────────────────────────────────
if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(
        f"Dataset not found at '{TRAIN_PATH}'. "
        "Download from Kaggle ('Housing Prices Competition') and place in data/."
    )

df = pd.read_csv(TRAIN_PATH)
df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
df["TotalFlrSF"] = df["1stFlrSF"] + df["2ndFlrSF"]

X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")


# ──────────────────────────────────────────────
# 3. Random Forest — hyperparameter tuning
# ──────────────────────────────────────────────
print("\n[1/4] Running RandomizedSearchCV for Random Forest ...")

param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 1.0],
}

rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE),
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    scoring="neg_mean_absolute_error",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf_search.fit(X_train, y_train)

best_rf = rf_search.best_estimator_
print(f"  Best params : {rf_search.best_params_}")
print(f"  Best CV MAE : {-rf_search.best_score_:,.2f}")


# ──────────────────────────────────────────────
# 4. Tuned RF — evaluation on test set
# ──────────────────────────────────────────────
print("\n[2/4] Evaluating tuned Random Forest on test set ...")

pred_rf = best_rf.predict(X_test)
mae_rf   = mean_absolute_error(y_test, pred_rf)
rmse_rf  = np.sqrt(mean_squared_error(y_test, pred_rf))
r2_rf    = r2_score(y_test, pred_rf)
print(f"  MAE  : ${mae_rf:,.2f}")
print(f"  RMSE : ${rmse_rf:,.2f}")
print(f"  R²   : {r2_rf:.4f}")


# ──────────────────────────────────────────────
# 5. Keras neural network — data prep & training
# ──────────────────────────────────────────────
print("\n[3/4] Training Keras neural network (comparison only) ...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

nn_model = Sequential([
    tf.keras.Input(shape=(X_train_scaled.shape[1],)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1),
])
nn_model.compile(optimizer="adam", loss="mae")

early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
nn_model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0,
)

pred_nn = nn_model.predict(X_test_scaled, verbose=0).flatten()
mae_nn   = mean_absolute_error(y_test, pred_nn)
rmse_nn  = np.sqrt(mean_squared_error(y_test, pred_nn))
r2_nn    = r2_score(y_test, pred_nn)
print(f"  MAE  : ${mae_nn:,.2f}")
print(f"  RMSE : ${rmse_nn:,.2f}")
print(f"  R²   : {r2_nn:.4f}")

nn_model.save(NN_PATH)
print(f"  Keras model saved → {NN_PATH} (reference only, not deployed)")


# ──────────────────────────────────────────────
# 6. Model selection & conditional export
# ──────────────────────────────────────────────
print(f"\n[4/4] Comparing tuned RF (MAE ${mae_rf:,.2f}) vs benchmark (MAE ${EXISTING_MAE_BENCHMARK:,.2f}) ...")

if mae_rf < EXISTING_MAE_BENCHMARK:
    joblib.dump(best_rf, MODEL_PATH)
    print(f"  ✅ Tuned RF is better — saved as {MODEL_PATH}")
else:
    print(f"  ⚠️  Tuned RF does NOT improve over existing model — {MODEL_PATH} unchanged")

# ──────────────────────────────────────────────
# 7. Summary
# ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("RESULTS SUMMARY")
print("=" * 50)
print(f"{'Model':<30} {'MAE':>10} {'RMSE':>10} {'R²':>8}")
print("-" * 60)
print(f"{'Tuned Random Forest':<30} ${mae_rf:>9,.0f} ${rmse_rf:>9,.0f} {r2_rf:>8.4f}")
print(f"{'Keras Neural Network':<30} ${mae_nn:>9,.0f} ${rmse_nn:>9,.0f} {r2_nn:>8.4f}")
print("=" * 50)
