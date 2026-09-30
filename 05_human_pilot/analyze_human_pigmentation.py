#!/usr/bin/env python3
"""Leakage-safe two-stage ML analysis (RF/GB screen-then-refine, nested CV)
applied to the 1000 Genomes EUR pigmentation pilot dataset built by
fetch_and_simulate_human_pigmentation.py.

Design mirrors the Arabidopsis synthetic-data pipeline
(../04_synthetic_benchmarks/run_synthetic_ft10_ml.py): marker screening is repeated fresh inside each
outer training fold (no marker selection before cross-validation), and the
causal-truth file is only consulted after all out-of-fold predictions and
importance rankings are finalized.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

OUT_DIR = Path(__file__).resolve().parent
GENOTYPE_DIR = OUT_DIR / "genotype_blocks"

OUTER_FOLDS = 5
STAGE1_RF_TREES = 300
STAGE1_GB_TREES = 150
STAGE2_RF_TREES = 500
STAGE2_GB_TREES = 300
TOP_K_PER_BLOCK_PER_MODEL = 10
TOP_N_REPORT = 50
BASE_SEED = 20260501


def make_stage1_models(seed: int) -> tuple[RandomForestRegressor, GradientBoostingRegressor]:
    rf = RandomForestRegressor(
        n_estimators=STAGE1_RF_TREES, max_features="sqrt", n_jobs=-1, random_state=seed
    )
    gb = GradientBoostingRegressor(
        n_estimators=STAGE1_GB_TREES, max_depth=3, learning_rate=0.05,
        subsample=0.8, max_features="sqrt", random_state=seed,
    )
    return rf, gb


def load_block_markers(block: dict) -> pd.DataFrame:
    marker_df = pd.read_csv(OUT_DIR / block["marker_map_path"])
    marker_df["global_marker_id"] = (
        "Chr" + marker_df["Chr"].astype(str) + "_" + marker_df["Pos"].astype(str)
    )
    return marker_df


def main() -> None:
    manifest = json.loads((OUT_DIR / "simulation_manifest.json").read_text())
    phenotype_df = pd.read_csv(OUT_DIR / "phenotypes.csv")
    y = phenotype_df["pigmentation_score_simulated"].to_numpy(dtype=float)
    blocks = manifest["blocks"]
    n_samples = len(y)

    folds = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=BASE_SEED)
    rf_oof = np.full(n_samples, np.nan)
    gb_oof = np.full(n_samples, np.nan)
    fold_rows = []
    rf_importance_sum: dict[str, float] = {}
    gb_importance_sum: dict[str, float] = {}
    rf_selected_folds: dict[str, int] = {}
    gb_selected_folds: dict[str, int] = {}

    for fold_number, (train_idx, test_idx) in enumerate(folds.split(y), 1):
        print(f"\n=== outer fold {fold_number}/{OUTER_FOLDS} ===", flush=True)
        rf_candidates: dict[str, float] = {}
        gb_candidates: dict[str, float] = {}

        for block in blocks:
            genotypes = np.load(OUT_DIR / block["genotype_path"])  # markers x samples
            marker_df = load_block_markers(block)
            train_x = genotypes[:, train_idx].T.astype(np.float64)
            rf, gb = make_stage1_models(
                BASE_SEED + 1000 * fold_number + hash(block["block_id"]) % 1000
            )
            rf.fit(train_x, y[train_idx])
            gb.fit(train_x, y[train_idx])

            for model_name, model, candidate_pool in [
                ("rf", rf, rf_candidates), ("gb", gb, gb_candidates)
            ]:
                importances = model.feature_importances_
                top_local = np.argsort(importances)[::-1][:TOP_K_PER_BLOCK_PER_MODEL]
                for local_i in top_local:
                    marker_id = marker_df.iloc[local_i]["global_marker_id"]
                    candidate_pool[marker_id] = importances[local_i]
            print(f"  screened block {block['block_id']} ({genotypes.shape[0]} markers)")

        all_candidate_ids = sorted(set(rf_candidates) | set(gb_candidates))
        print(f"  fold {fold_number}: {len(all_candidate_ids)} unique finalist markers")

        # Build a finalist feature matrix by re-reading only the needed rows.
        marker_lookup: dict[str, tuple[str, int]] = {}
        for block in blocks:
            marker_df = load_block_markers(block)
            for local_i, marker_id in enumerate(marker_df["global_marker_id"]):
                if marker_id in all_candidate_ids:
                    marker_lookup[marker_id] = (block["block_id"], local_i)

        finalist_matrix = np.zeros((n_samples, len(all_candidate_ids)), dtype=np.float64)
        block_cache: dict[str, np.ndarray] = {}
        for col, marker_id in enumerate(all_candidate_ids):
            block_id, local_i = marker_lookup[marker_id]
            if block_id not in block_cache:
                matching_block = next(b for b in blocks if b["block_id"] == block_id)
                block_cache[block_id] = np.load(OUT_DIR / matching_block["genotype_path"])
            finalist_matrix[:, col] = block_cache[block_id][local_i, :]

        x_train, x_test = finalist_matrix[train_idx], finalist_matrix[test_idx]
        rf_final = RandomForestRegressor(
            n_estimators=STAGE2_RF_TREES, max_features="sqrt", n_jobs=-1,
            random_state=BASE_SEED + 500_000 + fold_number,
        )
        gb_final = GradientBoostingRegressor(
            n_estimators=STAGE2_GB_TREES, max_depth=3, learning_rate=0.05,
            subsample=0.8, max_features="sqrt", random_state=BASE_SEED + 600_000 + fold_number,
        )
        rf_final.fit(x_train, y[train_idx])
        gb_final.fit(x_train, y[train_idx])
        rf_oof[test_idx] = rf_final.predict(x_test)
        gb_oof[test_idx] = gb_final.predict(x_test)

        for marker_id, importance in zip(all_candidate_ids, rf_final.feature_importances_):
            rf_importance_sum[marker_id] = rf_importance_sum.get(marker_id, 0.0) + importance
            rf_selected_folds[marker_id] = rf_selected_folds.get(marker_id, 0) + 1
        for marker_id, importance in zip(all_candidate_ids, gb_final.feature_importances_):
            gb_importance_sum[marker_id] = gb_importance_sum.get(marker_id, 0.0) + importance
            gb_selected_folds[marker_id] = gb_selected_folds.get(marker_id, 0) + 1

        fold_rows.append(
            {
                "fold": fold_number,
                "n_finalists": len(all_candidate_ids),
                "rf_r2_fold": float(r2_score(y[test_idx], rf_oof[test_idx])),
                "gb_r2_fold": float(r2_score(y[test_idx], gb_oof[test_idx])),
            }
        )
        print(f"  fold {fold_number}: RF R2={fold_rows[-1]['rf_r2_fold']:.3f}; "
              f"GB R2={fold_rows[-1]['gb_r2_fold']:.3f}")

    if np.isnan(rf_oof).any() or np.isnan(gb_oof).any():
        raise RuntimeError("Some samples never received out-of-fold predictions.")

    pd.DataFrame(fold_rows).to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    pd.DataFrame(
        {
            "sample": phenotype_df["sample"],
            "y_true": y,
            "rf_oof_pred": rf_oof,
            "gb_oof_pred": gb_oof,
        }
    ).to_csv(OUT_DIR / "out_of_fold_predictions.csv", index=False)

    all_marker_ids = sorted(set(rf_importance_sum) | set(gb_importance_sum))
    result_table = pd.DataFrame({"Marker": all_marker_ids})
    result_table["Chr"] = result_table["Marker"].str.extract(r"Chr(\d+)_").astype(int)
    result_table["Pos"] = result_table["Marker"].str.extract(r"_(\d+)$").astype(int)
    result_table["RF_mean_importance"] = [
        rf_importance_sum.get(m, 0.0) / OUTER_FOLDS for m in all_marker_ids
    ]
    result_table["GB_mean_importance"] = [
        gb_importance_sum.get(m, 0.0) / OUTER_FOLDS for m in all_marker_ids
    ]
    result_table["RF_selection_folds"] = [rf_selected_folds.get(m, 0) for m in all_marker_ids]
    result_table["GB_selection_folds"] = [gb_selected_folds.get(m, 0) for m in all_marker_ids]
    result_table["RF_rank"] = result_table["RF_mean_importance"].rank(ascending=False, method="min")
    result_table["GB_rank"] = result_table["GB_mean_importance"].rank(ascending=False, method="min")
    result_table.sort_values("RF_mean_importance", ascending=False).to_csv(
        OUT_DIR / "screened_candidates_and_importance.csv", index=False
    )

    top_rf = result_table.nlargest(TOP_N_REPORT, "RF_mean_importance").copy()
    top_gb = result_table.nlargest(TOP_N_REPORT, "GB_mean_importance").copy()
    top_rf.to_csv(OUT_DIR / f"top{TOP_N_REPORT}_random_forest.csv", index=False)
    top_gb.to_csv(OUT_DIR / f"top{TOP_N_REPORT}_gradient_boosting.csv", index=False)
    overlap = len(set(top_rf["Marker"]) & set(top_gb["Marker"]))

    # --- Post-hoc causal recovery check (loaded only now) -----------------------
    causal_df = pd.read_csv(OUT_DIR / "causal_truth_POSTHOC_ONLY.csv")
    causal_df["Marker"] = "Chr" + causal_df["chrom"].astype(str) + "_" + causal_df["pos"].astype(str)
    recovery_rows = []
    for _, causal in causal_df.iterrows():
        marker_id = causal["Marker"]
        matched = result_table[result_table["Marker"] == marker_id]
        for model_name, top_set, rank_col, imp_col, sel_col in [
            ("Random Forest", set(top_rf["Marker"]), "RF_rank", "RF_mean_importance", "RF_selection_folds"),
            ("Gradient Boosting", set(top_gb["Marker"]), "GB_rank", "GB_mean_importance", "GB_selection_folds"),
        ]:
            recovery_rows.append(
                {
                    "model": model_name,
                    "rsid": causal["rsid"],
                    "gene": causal["gene"],
                    "role": causal["role"],
                    "target_var_fraction": causal["target_var_fraction"],
                    "Marker": marker_id,
                    "final_importance_rank": (
                        int(matched.iloc[0][rank_col]) if not matched.empty else None
                    ),
                    "final_importance_value": (
                        float(matched.iloc[0][imp_col]) if not matched.empty else 0.0
                    ),
                    "selected_in_outer_training_folds": (
                        int(matched.iloc[0][sel_col]) if not matched.empty else 0
                    ),
                    "exactly_in_top50": marker_id in top_set,
                }
            )
    recovery_df = pd.DataFrame(recovery_rows)
    recovery_df.to_csv(OUT_DIR / "causal_recovery_posthoc.csv", index=False)

    summary = {
        "n_samples": n_samples,
        "n_markers_screened_per_fold": int(sum(b["n_markers"] for b in blocks)),
        "n_blocks": len(blocks),
        "outer_folds": OUTER_FOLDS,
        "rf_oof_r2": float(r2_score(y, rf_oof)),
        "gb_oof_r2": float(r2_score(y, gb_oof)),
        "rf_oof_rmse": float(math.sqrt(mean_squared_error(y, rf_oof))),
        "gb_oof_rmse": float(math.sqrt(mean_squared_error(y, gb_oof))),
        "rf_oof_mae": float(mean_absolute_error(y, rf_oof)),
        "gb_oof_mae": float(mean_absolute_error(y, gb_oof)),
        "top50_overlap": int(overlap),
        "candidates_in_importance_table": int(len(all_marker_ids)),
        "causal_loci_recovered_rf_top50": int(
            recovery_df[(recovery_df["model"] == "Random Forest") & recovery_df["exactly_in_top50"]].shape[0]
        ),
        "causal_loci_recovered_gb_top50": int(
            recovery_df[(recovery_df["model"] == "Gradient Boosting") & recovery_df["exactly_in_top50"]].shape[0]
        ),
        "method": (
            "Fold-specific per-block screening followed by joint finalist-model "
            "fitting; screening and fitting use outer-training samples only."
        ),
    }
    (OUT_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
