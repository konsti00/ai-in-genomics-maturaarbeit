#!/usr/bin/env python3
"""Repeat the blind and context-calibrated FT10 simulations across multiple
random seeds, to see how much the single-seed results in
``../blind_simulation`` and ``../context_calibrated_simulation`` could have
varied by chance.

This reuses the exact generation and leakage-safe evaluation logic from
``../run_synthetic_ft10_ml.py`` (imported as a module, not copied), so the
simulation design, causal-marker placement rules, and nested
screen-then-refine cross-validation are identical to the earlier single-seed
runs.

Two settings are deliberately reduced from the earlier single-seed runs, to
keep 10 total runs (5 seeds x 2 scenarios) computationally feasible in one
session:
  - Outer cross-validation folds: 3 instead of 5.
  - Screening/finalist tree counts: half of the original Stage-1 and Stage-2
    settings.

Both changes are documented in MULTI_SEED_REPORT.md. They make this a
lighter-weight repeatability check, not a like-for-like replication of the
single-seed numbers already reported.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

import run_synthetic_ft10_ml as core  # noqa: E402

N_SEEDS_PER_SCENARIO = 5
OUTER_FOLDS = 3
STAGE1_RF_TREES = 150
STAGE1_GB_TREES = 75
STAGE2_RF_TREES = 300
STAGE2_GB_TREES = 150
BASE_MULTI_SEED = 20270101
SCENARIO_SEED_OFFSET = {"blind": 0, "context_calibrated": 100_000}


def run_one(scenario: str, seed_index: int) -> dict:
    seed = BASE_MULTI_SEED + SCENARIO_SEED_OFFSET[scenario] + seed_index
    run_name = f"{scenario}_seed{seed_index + 1}"
    out_dir = THIS_DIR / run_name
    start = time.time()
    print(f"\n=== {run_name} (seed {seed}) ===", flush=True)

    core.generate_genotypes_and_phenotypes(scenario, out_dir, seed_override=seed)
    core.analyze_dataset(
        out_dir,
        seed_override=seed,
        outer_folds_override=OUTER_FOLDS,
        stage1_rf_trees=STAGE1_RF_TREES,
        stage1_gb_trees=STAGE1_GB_TREES,
        stage2_rf_trees=STAGE2_RF_TREES,
        stage2_gb_trees=STAGE2_GB_TREES,
    )
    core.write_prior_locus_comparison(out_dir)

    summary = json.loads((out_dir / "analysis_summary.json").read_text())
    recovery = pd.read_csv(out_dir / "causal_recovery_posthoc.csv")
    loci = pd.read_csv(out_dir / "prior_real_loci_comparison_POSTHOC_ONLY.csv")
    recovery_by_model = recovery.groupby("model")["exactly_in_top50"].sum().to_dict()

    # Free disk space: the genotype matrices are large and are not needed
    # once cross-validated predictions, importances, and recovery files are
    # written. Manifest, phenotypes, causal truth, and all result tables are
    # kept.
    genotype_dir = out_dir / "genotype_blocks"
    if genotype_dir.exists():
        shutil.rmtree(genotype_dir)

    elapsed_minutes = (time.time() - start) / 60.0
    row = {
        "scenario": scenario,
        "seed_index": seed_index + 1,
        "random_seed": seed,
        "run_name": run_name,
        "rf_oof_r2": summary["rf_oof_r2"],
        "gb_oof_r2": summary["gb_oof_r2"],
        "rf_oof_rmse": summary["rf_oof_rmse"],
        "gb_oof_rmse": summary["gb_oof_rmse"],
        "top50_overlap": summary["top50_overlap"],
        "rf_causal_in_top50": int(recovery_by_model.get("Random Forest", 0)),
        "gb_causal_in_top50": int(recovery_by_model.get("Gradient Boosting", 0)),
        "n_causal_markers": int(recovery["causal_marker"].nunique()),
        "any_prior_locus_overlap": bool(
            loci["top50_markers_in_window"].sum() > 0
            or loci["planted_causal_markers_in_window"].sum() > 0
        ),
        "elapsed_minutes": round(elapsed_minutes, 2),
    }
    print(f"=== finished {run_name} in {elapsed_minutes:.1f} min: {row} ===", flush=True)
    return row


def main() -> None:
    rows = []
    for scenario in ["blind", "context_calibrated"]:
        for seed_index in range(N_SEEDS_PER_SCENARIO):
            rows.append(run_one(scenario, seed_index))
            # Persist progress after every run in case of interruption.
            pd.DataFrame(rows).to_csv(THIS_DIR / "per_run_results.csv", index=False)

    results = pd.DataFrame(rows)
    results.to_csv(THIS_DIR / "per_run_results.csv", index=False)

    aggregate_rows = []
    for scenario, group in results.groupby("scenario"):
        for model, r2_col, causal_col, rmse_col in [
            ("Random Forest", "rf_oof_r2", "rf_causal_in_top50", "rf_oof_rmse"),
            ("Gradient Boosting", "gb_oof_r2", "gb_causal_in_top50", "gb_oof_rmse"),
        ]:
            aggregate_rows.append(
                {
                    "scenario": scenario,
                    "model": model,
                    "n_seeds": int(len(group)),
                    "mean_oof_r2": float(group[r2_col].mean()),
                    "sd_oof_r2": float(group[r2_col].std(ddof=1)),
                    "min_oof_r2": float(group[r2_col].min()),
                    "max_oof_r2": float(group[r2_col].max()),
                    "mean_oof_rmse": float(group[rmse_col].mean()),
                    "sd_oof_rmse": float(group[rmse_col].std(ddof=1)),
                    "mean_causal_in_top50": float(group[causal_col].mean()),
                    "sd_causal_in_top50": float(group[causal_col].std(ddof=1)),
                    "min_causal_in_top50": int(group[causal_col].min()),
                    "max_causal_in_top50": int(group[causal_col].max()),
                    "n_causal_markers": int(group["n_causal_markers"].iloc[0]),
                    "any_prior_locus_overlap_in_any_seed": bool(
                        group["any_prior_locus_overlap"].any()
                    ),
                }
            )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(THIS_DIR / "aggregate_results.csv", index=False)

    top50_overlap_agg = (
        results.groupby("scenario")["top50_overlap"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    top50_overlap_agg.to_csv(THIS_DIR / "top50_overlap_by_scenario.csv", index=False)

    print("\n\n=== ALL RUNS COMPLETE ===")
    print(results.to_string(index=False))
    print("\n=== AGGREGATE (mean +/- SD across seeds) ===")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
