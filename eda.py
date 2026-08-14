"""
eda.py
======
Iowa Housing ML Pipeline — Exploratory Data Analysis (Person 1)
---------------------------------------------------------------
Purpose:
    Loads the Kaggle Iowa housing dataset and produces a suite of
    diagnostic plots saved to the outputs/ directory.

Usage:
    python eda.py

Outputs (saved to outputs/):
    - missing_values.png
    - saleprice_distribution.png
    - correlation_matrix.png
    - top_features_vs_target.png
    - outliers_boxplot.png
    - saleprice_by_<feature>.png  (for Neighborhood, OverallQual, KitchenQual)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 100)

TRAIN_PATH = os.path.join("data", "train.csv")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────
try:
    df = pd.read_csv(TRAIN_PATH)
except FileNotFoundError:
    print(
        "Error: data/train.csv not found. "
        "Download from Kaggle ('Housing Prices Competition') and place in data/."
    )
    raise SystemExit(1)


# ──────────────────────────────────────────────
# 2. Basic overview
# ──────────────────────────────────────────────
print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"Shape       : {df.shape}")
print(f"Numeric cols: {df.select_dtypes(include=[np.number]).shape[1]}")
print(f"String cols : {df.select_dtypes(include='object').shape[1]}")
print(f"\nDescriptive stats (numeric):")
print(df.describe().T.to_string())


# ──────────────────────────────────────────────
# 3. Missing values
# ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)

missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = (
    pd.DataFrame({"count": missing, "pct": missing_pct})
    .query("count > 0")
    .sort_values("pct", ascending=False)
)
print(missing_df.to_string())
print(f"\nDuplicate rows : {df.duplicated().sum()}")
print(f"Unique Id      : {df['Id'].is_unique}")

plt.figure(figsize=(10, 8))
sns.barplot(x=missing_df["pct"], y=missing_df.index)
plt.title("Missing values by column (%)")
plt.xlabel("% Missing")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "missing_values.png"), dpi=150)
plt.close()


# ──────────────────────────────────────────────
# 4. SalePrice distribution
# ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("SALEPRICE DISTRIBUTION")
print("=" * 50)
print(f"Skewness (raw)    : {df['SalePrice'].skew():.4f}")
print(f"Skewness (log+1)  : {np.log1p(df['SalePrice']).skew():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df["SalePrice"], kde=True, ax=axes[0])
axes[0].set_title("SalePrice distribution")
sns.histplot(np.log1p(df["SalePrice"]), kde=True, ax=axes[1])
axes[1].set_title("log(SalePrice) distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "saleprice_distribution.png"), dpi=150)
plt.close()


# ──────────────────────────────────────────────
# 5. Correlation matrix
# ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("CORRELATION WITH SALEPRICE (Top 15 / Bottom 10)")
print("=" * 50)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr = df[numeric_cols].corr()
corr_target = corr["SalePrice"].sort_values(ascending=False)
print("\nTop 15 positive:\n", corr_target.head(15).to_string())
print("\nTop 10 negative:\n", corr_target.tail(10).to_string())

plt.figure(figsize=(14, 12))
sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.3)
plt.title("Correlation matrix (numeric features)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_matrix.png"), dpi=150)
plt.close()


# ──────────────────────────────────────────────
# 6. Top features vs target
# ──────────────────────────────────────────────
top_features = corr_target.index[1:6]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, feat in zip(axes.flatten()[:5], top_features):
    sns.scatterplot(x=df[feat], y=df["SalePrice"], ax=ax, alpha=0.5)
    ax.set_title(f"{feat} vs SalePrice")
axes.flatten()[5].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_features_vs_target.png"), dpi=150)
plt.close()


# ──────────────────────────────────────────────
# 7. Outlier inspection
# ──────────────────────────────────────────────
outliers = df[(df["GrLivArea"] > 4000) & (df["SalePrice"] < 300_000)]
print(f"\nPotential outliers (GrLivArea>4000 & SalePrice<$300k): {len(outliers)}")
if not outliers.empty:
    print(outliers[["Id", "GrLivArea", "SalePrice"]].to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, feat in zip(axes, ["GrLivArea", "LotArea", "TotalBsmtSF"]):
    sns.boxplot(y=df[feat], ax=ax)
    ax.set_title(feat)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "outliers_boxplot.png"), dpi=150)
plt.close()


# ──────────────────────────────────────────────
# 8. Categorical feature analysis
# ──────────────────────────────────────────────
for feat in ["Neighborhood", "OverallQual", "KitchenQual"]:
    order = df.groupby(feat)["SalePrice"].median().sort_values(ascending=False).index
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=feat, y="SalePrice", data=df, order=order)
    plt.xticks(rotation=45)
    plt.title(f"SalePrice by {feat}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"saleprice_by_{feat}.png"), dpi=150)
    plt.close()


print(f"\n✅  EDA complete. All plots saved to '{OUTPUT_DIR}/'.")
