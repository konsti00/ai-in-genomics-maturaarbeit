"""
AI-analysis (machine-learning) genotype-to-phenotype mapping for FT10.

Deliberately does NOT use any GWAS statistical model:
- No mixed linear model
- No kinship/relatedness correction term
- No per-SNP hypothesis test / p-value
- No GWAS software or plugin (GAPIT, TASSEL, PLINK, GEMMA, etc.)

Instead, two general-purpose supervised machine-learning regressors are
trained to predict the FT10 phenotype directly from the genotype matrix:
  1. Random Forest Regressor
  2. Gradient Boosting Regressor

"Importance" of a SNP is derived from each model's own internal feature
importance measure (impurity-based for both, forest-native), not from any
statistical association test. Predictive performance is assessed with
k-fold cross-validation (R^2), which has no equivalent concept in classical
GWAS.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score

BASE = Path(__file__).resolve().parent
GENOTYPE_FILE = BASE / "ml_genotype_matrix.npz"
MARKER_FILE = BASE / "ml_marker_map.csv"
PHENOTYPE_FILE = BASE / "ml_phenotype.csv"

RANDOM_STATE = 42
N_FOLDS = 5


def main():
    data = np.load(GENOTYPE_FILE, allow_pickle=True)
    X = data["X"]
    taxa = data["taxa"]
    markers = data["markers"]

    phenotype = pd.read_csv(PHENOTYPE_FILE)
    phenotype["Taxa"] = phenotype["Taxa"].astype(str)
    phenotype = phenotype.set_index("Taxa").loc[taxa]
    y = phenotype["FT10"].to_numpy(dtype=float)

    print(f"Genotype matrix: {X.shape[0]} samples x {X.shape[1]} markers")
    print(f"Phenotype range: {y.min():.1f} - {y.max():.1f}, mean {y.mean():.1f}")

    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = {}

    # ---- Random Forest ----
    print("\nTraining Random Forest (cross-validated)...")
    rf = RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf_pred = cross_val_predict(rf, X, y, cv=kfold, n_jobs=1)
    rf_r2 = r2_score(y, rf_pred)
    print(f"Random Forest cross-validated R^2: {rf_r2:.4f}")

    rf.fit(X, y)
    rf_importance = rf.feature_importances_
    results["RandomForest"] = rf_importance

    # ---- Gradient Boosting ----
    print("\nTraining Gradient Boosting (cross-validated)...")
    gb = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        max_features="sqrt",
        random_state=RANDOM_STATE,
    )
    gb_pred = cross_val_predict(gb, X, y, cv=kfold, n_jobs=1)
    gb_r2 = r2_score(y, gb_pred)
    print(f"Gradient Boosting cross-validated R^2: {gb_r2:.4f}")

    gb.fit(X, y)
    gb_importance = gb.feature_importances_
    results["GradientBoosting"] = gb_importance

    # ---- Combine into one ranked table ----
    marker_map = pd.read_csv(MARKER_FILE).set_index("Marker").loc[markers].reset_index()
    marker_map["RF_importance"] = results["RandomForest"]
    marker_map["GB_importance"] = results["GradientBoosting"]
    marker_map["RF_rank"] = marker_map["RF_importance"].rank(ascending=False, method="min")
    marker_map["GB_rank"] = marker_map["GB_importance"].rank(ascending=False, method="min")

    marker_map = marker_map.sort_values("RF_importance", ascending=False)
    marker_map.to_csv(BASE / "ml_feature_importance_all.csv", index=False)

    top_rf = marker_map.sort_values("RF_importance", ascending=False).head(50)
    top_gb = marker_map.sort_values("GB_importance", ascending=False).head(50)
    top_rf.to_csv(BASE / "ml_top50_random_forest.csv", index=False)
    top_gb.to_csv(BASE / "ml_top50_gradient_boosting.csv", index=False)

    top_rf_set = set(top_rf["Marker"])
    top_gb_set = set(top_gb["Marker"])
    overlap = top_rf_set & top_gb_set

    summary = [
        f"Samples: {X.shape[0]}",
        f"Markers analyzed: {X.shape[1]:,}",
        f"Random Forest cross-validated R^2: {rf_r2:.4f}",
        f"Gradient Boosting cross-validated R^2: {gb_r2:.4f}",
        f"Overlap between RF top-50 and GB top-50: {len(overlap)} / 50",
        "",
        "Top 10 SNPs by Random Forest importance:",
    ]
    for _, row in top_rf.head(10).iterrows():
        summary.append(
            f"  {row['Marker']} (Chr {int(row['Chr'])}, pos {int(row['Pos'])}) "
            f"RF={row['RF_importance']:.5f} GB_rank={int(row['GB_rank'])}"
        )
    summary.append("")
    summary.append("Top 10 SNPs by Gradient Boosting importance:")
    for _, row in top_gb.head(10).iterrows():
        summary.append(
            f"  {row['Marker']} (Chr {int(row['Chr'])}, pos {int(row['Pos'])}) "
            f"GB={row['GB_importance']:.5f} RF_rank={int(row['RF_rank'])}"
        )

    (BASE / "ml_analysis_summary.txt").write_text("\n".join(summary) + "\n")
    print("\n" + "\n".join(summary))
    print(
        "\nSaved ml_feature_importance_all.csv, ml_top50_random_forest.csv, "
        "ml_top50_gradient_boosting.csv, ml_analysis_summary.txt"
    )


if __name__ == "__main__":
    main()
