"""
AI-analysis (machine-learning) genotype-to-phenotype mapping for FT10.
FULL-RESOLUTION, memory-safe, two-stage version (no thinning of SNPs).

Deliberately does NOT use any GWAS statistical model:
- No mixed linear model, no kinship/relatedness correction term
- No per-SNP hypothesis test / p-value
- No GWAS software or plugin (GAPIT, TASSEL, PLINK, GEMMA, etc.)

Two general-purpose supervised ML regressors are used throughout:
  1. Random Forest Regressor
  2. Gradient Boosting Regressor

Why two stages (this is the memory-safe adaptation for all 3,557,529 SNPs):
- Loading all 3.56M SNPs x 1,003 samples into a single scikit-learn model
  caused severe memory thrashing on this machine (17GB RAM) and made no
  reliable forward progress.
- STAGE 1 (screening): train a fresh RF + GB pair independently on each of
  the 108 existing genotype blocks (~30-40k markers each, the same blocks
  used for the classical GWAS run, at full resolution / no thinning) and
  record each model's own feature importance for every SNP. This still
  evaluates every single filtered SNP in the dataset -- nothing is thinned
  or pre-selected by association -- it is purely an engineering split to
  keep each individual model fit within memory.
- STAGE 2 (joint refinement): pool the top-ranked SNPs from stage 1 across
  all blocks and both models into one manageable joint matrix, then train
  one final RF and one final GB on that combined set. This stage captures
  interactions between top markers from different genomic blocks (which
  stage 1 cannot see) and produces one overall cross-validated R^2 for each
  model, analogous to a final predictive-performance number.

This two-stage "screen then refine" design keeps each individual model fit
within the available memory when the number of markers far exceeds the
number of samples.

Limitation: the Stage-2 candidates are selected using all samples before
cross-validation, so the reported cross-validated R^2 is exploratory and may
be optimistic. The later synthetic and human analyses therefore repeat the
screening inside every training fold (nested cross-validation).
"""

from pathlib import Path
import re
import gc

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score

BASE = Path(__file__).resolve().parent
SOURCE_BASE = BASE.parents[1] / "02_arabidopsis_gwas"
BLOCK_DIR = SOURCE_BASE / "ft10_blocks"
PHENOTYPE_FILE = BASE / "ml_phenotype.csv"

RANDOM_STATE = 42
N_FOLDS = 5
STAGE1_N_ESTIMATORS_RF = 300
STAGE1_N_ESTIMATORS_GB = 150
TOP_K_PER_BLOCK = 25  # top markers kept per block per model, going into stage 2
STAGE2_N_ESTIMATORS_RF = 500
STAGE2_N_ESTIMATORS_GB = 300


def block_sort_key(path):
    match = re.search(r"chr(\d+)_block(\d+)", path.name)
    return tuple(map(int, match.groups())) if match else (999, 999)


def stage1_screen(taxa_order, y):
    block_files = sorted(BLOCK_DIR.glob("*.csv.gz"), key=block_sort_key)
    if not block_files:
        raise FileNotFoundError(f"No genotype blocks found in {BLOCK_DIR}")

    all_importance = []

    for i, block_file in enumerate(block_files):
        block_name = block_file.name.replace(".csv.gz", "")
        marker_file = BLOCK_DIR / f"{block_name}_markers.csv"

        raw = pd.read_csv(block_file)
        raw = raw.set_index("Marker")
        raw = raw.reindex(columns=taxa_order)
        X_block = raw.T.to_numpy(dtype=np.int8)

        markers = pd.read_csv(marker_file).set_index("Marker").loc[raw.index]

        rf = RandomForestRegressor(
            n_estimators=STAGE1_N_ESTIMATORS_RF,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        rf.fit(X_block, y)
        rf_importance = rf.feature_importances_

        gb = GradientBoostingRegressor(
            n_estimators=STAGE1_N_ESTIMATORS_GB,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            max_features="sqrt",
            random_state=RANDOM_STATE,
        )
        gb.fit(X_block, y)
        gb_importance = gb.feature_importances_

        block_table = markers.reset_index()
        block_table["RF_importance"] = rf_importance
        block_table["GB_importance"] = gb_importance
        block_table["source_block"] = block_name
        all_importance.append(block_table)

        print(
            f"[{i+1}/{len(block_files)}] {block_name}: {X_block.shape[1]} markers screened",
            flush=True,
        )

        del raw, X_block, rf, gb
        gc.collect()

    combined = pd.concat(all_importance, ignore_index=True)
    return combined


def stage2_refine(combined, taxa_order, y):
    n_blocks = combined["source_block"].nunique()
    top_rf = combined.sort_values("RF_importance", ascending=False).head(
        TOP_K_PER_BLOCK * n_blocks
    )
    top_gb = combined.sort_values("GB_importance", ascending=False).head(
        TOP_K_PER_BLOCK * n_blocks
    )
    finalist_markers = pd.unique(pd.concat([top_rf["Marker"], top_gb["Marker"]]))
    print(f"Stage 2 finalist marker pool: {len(finalist_markers)} unique SNPs", flush=True)

    finalist_set = set(finalist_markers)
    block_files = sorted(BLOCK_DIR.glob("*.csv.gz"), key=block_sort_key)
    columns = []
    for block_file in block_files:
        block_name = block_file.name.replace(".csv.gz", "")
        block_markers_in_pool = combined.loc[
            (combined["source_block"] == block_name)
            & (combined["Marker"].isin(finalist_set)),
            "Marker",
        ]
        if block_markers_in_pool.empty:
            continue
        raw = pd.read_csv(block_file)
        raw = raw.set_index("Marker")
        raw = raw.reindex(columns=taxa_order)
        subset = raw.loc[raw.index.intersection(block_markers_in_pool)]
        columns.append(subset.T.astype(np.int8))

    X_final = pd.concat(columns, axis=1)
    X_final = X_final.loc[:, ~X_final.columns.duplicated()]
    X = X_final.to_numpy(dtype=np.int8)
    marker_names = np.array(X_final.columns)

    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rf = RandomForestRegressor(
        n_estimators=STAGE2_N_ESTIMATORS_RF,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf_pred = cross_val_predict(rf, X, y, cv=kfold, n_jobs=1)
    rf_r2 = r2_score(y, rf_pred)
    rf.fit(X, y)

    gb = GradientBoostingRegressor(
        n_estimators=STAGE2_N_ESTIMATORS_GB,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        max_features="sqrt",
        random_state=RANDOM_STATE,
    )
    gb_pred = cross_val_predict(gb, X, y, cv=kfold, n_jobs=1)
    gb_r2 = r2_score(y, gb_pred)
    gb.fit(X, y)

    final_table = pd.DataFrame({"Marker": marker_names})
    final_table["RF_final_importance"] = rf.feature_importances_
    final_table["GB_final_importance"] = gb.feature_importances_
    return final_table, rf_r2, gb_r2


def main():
    phenotype = pd.read_csv(PHENOTYPE_FILE)
    phenotype["Taxa"] = phenotype["Taxa"].astype(str)
    taxa_order = phenotype["Taxa"].tolist()
    y = phenotype.set_index("Taxa").loc[taxa_order, "FT10"].to_numpy(dtype=float)

    print("=== STAGE 1: full-resolution per-block screening (all SNPs, no thinning) ===", flush=True)
    combined = stage1_screen(taxa_order, y)
    combined["RF_rank"] = combined["RF_importance"].rank(ascending=False, method="min")
    combined["GB_rank"] = combined["GB_importance"].rank(ascending=False, method="min")
    combined = combined.sort_values("RF_importance", ascending=False)
    combined.to_csv(BASE / "ml_feature_importance_all.csv", index=False)
    print(f"Stage 1 complete: {len(combined):,} SNPs screened across all blocks", flush=True)

    print("\n=== STAGE 2: joint refinement model on pooled top candidates ===", flush=True)
    final_table, rf_r2, gb_r2 = stage2_refine(combined, taxa_order, y)
    marker_map = combined[["Marker", "Chr", "Pos"]].drop_duplicates("Marker")
    final_table = final_table.merge(marker_map, on="Marker", how="left")
    final_table = final_table.sort_values("RF_final_importance", ascending=False)

    top_rf = final_table.sort_values("RF_final_importance", ascending=False).head(50)
    top_gb = final_table.sort_values("GB_final_importance", ascending=False).head(50)
    top_rf.to_csv(BASE / "ml_top50_random_forest.csv", index=False)
    top_gb.to_csv(BASE / "ml_top50_gradient_boosting.csv", index=False)

    overlap = set(top_rf["Marker"]) & set(top_gb["Marker"])

    summary = [
        "FULL-RESOLUTION (no thinning) two-stage AI-analysis",
        f"Samples: {len(taxa_order)}",
        f"Markers screened in Stage 1 (all filtered SNPs, no thinning): {len(combined):,}",
        f"Stage 2 joint refinement model size: {len(final_table):,} pooled top candidate SNPs",
        f"Stage 2 Random Forest cross-validated R^2: {rf_r2:.4f}",
        f"Stage 2 Gradient Boosting cross-validated R^2: {gb_r2:.4f}",
        f"Overlap between RF top-50 and GB top-50 (stage 2): {len(overlap)} / 50",
        "",
        "Top 10 SNPs by Stage-2 Random Forest importance:",
    ]
    for _, row in top_rf.head(10).iterrows():
        summary.append(
            f"  {row['Marker']} (Chr {int(row['Chr'])}, pos {int(row['Pos'])}) "
            f"RF={row['RF_final_importance']:.5f}"
        )
    summary.append("")
    summary.append("Top 10 SNPs by Stage-2 Gradient Boosting importance:")
    for _, row in top_gb.head(10).iterrows():
        summary.append(
            f"  {row['Marker']} (Chr {int(row['Chr'])}, pos {int(row['Pos'])}) "
            f"GB={row['GB_final_importance']:.5f}"
        )

    (BASE / "ml_analysis_summary.txt").write_text("\n".join(summary) + "\n")
    print("\n" + "\n".join(summary))
    print(
        "\nSaved ml_feature_importance_all.csv (stage 1, all SNPs), "
        "ml_top50_random_forest.csv, ml_top50_gradient_boosting.csv "
        "(stage 2 final), ml_analysis_summary.txt"
    )


if __name__ == "__main__":
    main()
