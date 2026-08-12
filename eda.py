import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 100)

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

try:
    df = pd.read_csv("data/train.csv")
except FileNotFoundError:
    print("Error: data/train.csv nuk u gjet. Ju lutem shkarkoni nga Kaggle dhe vendoseni në folderin 'data/'.")
    exit(1)

print("--- 2. Shape, Tipet, Statistikat ---")
print("Shape:", df.shape)
print("\nHead:\n", df.head())

print("\nTipet e variablave:")
print(df.dtypes.value_counts())
print("\nInfo:")
df.info()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

print("\nNumerike:", len(numeric_cols))
print("Kategorike:", len(categorical_cols))

print("\nStatistika përshkruese (Numerike):")
print(df[numeric_cols].describe().T)
print("\nStatistika përshkruese (Kategorike):")
print(df[categorical_cols].describe().T)

print("\n--- 3. Missing Values ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100

missing_summary = pd.DataFrame({
    "missing_count": missing,
    "missing_pct": missing_pct
}).sort_values("missing_pct", ascending=False)

missing_summary = missing_summary[missing_summary["missing_count"] > 0]
print(missing_summary)

plt.figure(figsize=(10, 8))
sns.barplot(x=missing_summary["missing_pct"], y=missing_summary.index)
plt.title("Përqindja e vlerave që mungojnë sipas kolonave")
plt.xlabel("% Missing")
plt.tight_layout()
plt.savefig("outputs/missing_values.png", dpi=150)
plt.close()

print("\nDuplicate rows:", df.duplicated().sum())
print("Id unik:", df["Id"].is_unique)

print("\n--- 4. Histogrami i SalePrice + Matrica e Korrelacionit ---")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df["SalePrice"], kde=True, ax=axes[0])
axes[0].set_title("Shpërndarja e SalePrice")

sns.histplot(np.log1p(df["SalePrice"]), kde=True, ax=axes[1])
axes[1].set_title("Shpërndarja e log(SalePrice)")

plt.tight_layout()
plt.savefig("outputs/saleprice_distribution.png", dpi=150)
plt.close()

print("Skewness (SalePrice):", df["SalePrice"].skew())
print("Skewness (log SalePrice):", np.log1p(df["SalePrice"]).skew())

plt.figure(figsize=(14, 12))
corr = df[numeric_cols].corr()
sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.3)
plt.title("Matrica e korrelacionit (features numerike)")
plt.tight_layout()
plt.savefig("outputs/correlation_matrix.png", dpi=150)
plt.close()

corr_target = corr["SalePrice"].sort_values(ascending=False)
print("\nTop pozitive (15):")
print(corr_target.head(15))
print("\nTop negative (10):")
print(corr_target.tail(10))

print("\n--- 5. Shtesa të Rekomanduara ---")
top_features = corr_target.index[1:6]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, feat in zip(axes.flatten()[:5], top_features):
    sns.scatterplot(x=df[feat], y=df["SalePrice"], ax=ax, alpha=0.5)
    ax.set_title(f"{feat} vs SalePrice")
axes.flatten()[5].set_visible(False) # hide empty 6th subplot
plt.tight_layout()
plt.savefig("outputs/top_features_vs_target.png", dpi=150)
plt.close()

outliers = df[(df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)]
print("\nOutliers:")
print(outliers[["Id", "GrLivArea", "SalePrice"]])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, feat in zip(axes, ["GrLivArea", "LotArea", "TotalBsmtSF"]):
    sns.boxplot(y=df[feat], ax=ax)
    ax.set_title(feat)
plt.tight_layout()
plt.savefig("outputs/outliers_boxplot.png", dpi=150)
plt.close()

for feat in ["Neighborhood", "OverallQual", "KitchenQual"]:
    plt.figure(figsize=(10, 5))
    order = df.groupby(feat)["SalePrice"].median().sort_values(ascending=False).index
    sns.boxplot(x=feat, y="SalePrice", data=df, order=order)
    plt.xticks(rotation=45)
    plt.title(f"SalePrice sipas {feat}")
    plt.tight_layout()
    plt.savefig(f"outputs/saleprice_by_{feat}.png", dpi=150)
    plt.close()

print("\nSkripti përfundoi! Grafikët janë ruajtur në 'outputs/'.")
