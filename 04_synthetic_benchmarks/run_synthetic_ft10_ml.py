#!/usr/bin/env python3
"""Generate and analyze two synthetic FT10 genotype/phenotype datasets.

The model pipeline uses only general-purpose Random Forest and Gradient
Boosting regressors. All feature screening is repeated within each outer
training fold, so the held-out fold does not influence marker selection.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.special import ndtri
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT.parent / "02_arabidopsis_gwas"
N_SAMPLES = 1003
N_MARKERS = 3_557_529
CHROMOSOMES = {
    1: {"length_bp": 30_427_613, "markers": 865_603, "blocks": 26},
    2: {"length_bp": 19_697_717, "markers": 600_877, "blocks": 19},
    3: {"length_bp": 23_459_627, "markers": 722_432, "blocks": 22},
    4: {"length_bp": 18_585_004, "markers": 584_671, "blocks": 18},
    5: {"length_bp": 26_975_377, "markers": 783_946, "blocks": 23},
}
MAF_BINS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
MAF_COUNTS = [810_093, 945_522, 590_076, 510_193, 293_428, 218_872, 189_345]
LD_DISTANCE_SUMMARY = [
    {"distance_bp": "0-1 kb", "median_r2": 0.0119, "pairs_sampled": 68_478},
    {"distance_bp": "1-10 kb", "median_r2": 0.0087, "pairs_sampled": 226_004},
    {"distance_bp": "10-50 kb", "median_r2": 0.0051, "pairs_sampled": 10_893},
]

N_CAUSAL = 20
TARGET_LATENT_SIGNAL_FRACTION = 0.5
OUTER_FOLDS = 5
TOP_K_PER_BLOCK_PER_MODEL = 10
STAGE1_RF_TREES = 300
STAGE1_GB_TREES = 150
STAGE2_RF_TREES = 500
STAGE2_GB_TREES = 300
BASE_SEED = 20260928


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def marker_label(chromosome: int, chromosome_index: int) -> str:
    return f"Chr{chromosome}_sim_{chromosome_index + 1:07d}"


def marker_position(chromosome: int, chromosome_index: int) -> int:
    info = CHROMOSOMES[chromosome]
    return int((chromosome_index + 0.5) * info["length_bp"] / info["markers"]) + 1


def sample_context_maf(rng: np.random.Generator, size: int) -> np.ndarray:
    counts = np.asarray(MAF_COUNTS, dtype=np.float64)
    probabilities = counts / counts.sum()
    bin_ids = rng.choice(len(MAF_BINS) - 1, size=size, p=probabilities)
    left = np.asarray(MAF_BINS[:-1])[bin_ids]
    right = np.asarray(MAF_BINS[1:])[bin_ids]
    return rng.uniform(left, right).astype(np.float32)


def generate_genotypes_and_phenotypes(
    scenario: str, out_dir: Path, *, seed_override: int | None = None
) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} already contains files. Refusing to overwrite prior results."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    genotype_dir = out_dir / "genotype_blocks"
    genotype_dir.mkdir()

    if seed_override is not None:
        seed = seed_override
    else:
        seed = BASE_SEED + (0 if scenario == "blind" else 1)
    rng = np.random.default_rng(seed)
    phenotype_reference: np.ndarray | None = None
    if scenario == "context_calibrated":
        phenotype_path = SOURCE_DIR / "ft10_gapit_phenotype.csv"
        observed = pd.read_csv(phenotype_path)["FT10"].dropna().to_numpy(dtype=float)
        if len(observed) != N_SAMPLES:
            raise ValueError(
                f"Expected {N_SAMPLES} FT10 values in {phenotype_path}; got {len(observed)}."
            )
        phenotype_reference = np.sort(observed)

    maf_by_chromosome: dict[int, np.ndarray] = {}
    offsets: dict[int, int] = {}
    cumulative = 0
    causal_global_ids: list[int] = []
    causal_mafs: list[float] = []
    for chromosome, info in CHROMOSOMES.items():
        offsets[chromosome] = cumulative
        count = info["markers"]
        if scenario == "blind":
            maf = rng.uniform(0.05, 0.5, size=count).astype(np.float32)
        else:
            maf = sample_context_maf(rng, count)
        maf_by_chromosome[chromosome] = maf
        cumulative += count
    if cumulative != N_MARKERS:
        raise ValueError(f"Chromosome counts total {cumulative:,}, expected {N_MARKERS:,}.")

    eligible_global = np.concatenate(
        [
            offsets[chromosome] + np.flatnonzero(maf >= 0.05)
            for chromosome, maf in maf_by_chromosome.items()
        ]
    )
    causal_global_ids = np.sort(
        rng.choice(eligible_global, size=N_CAUSAL, replace=False)
    ).tolist()
    global_to_causal = {global_id: index for index, global_id in enumerate(causal_global_ids)}
    causal_dosages = np.empty((N_SAMPLES, N_CAUSAL), dtype=np.float32)
    causal_mafs = np.empty(N_CAUSAL, dtype=np.float32)
    causal_observed = np.zeros(N_CAUSAL, dtype=bool)

    blocks: list[dict[str, object]] = []
    sample_ids = [f"SYN{i:04d}" for i in range(1, N_SAMPLES + 1)]
    global_offset = 0
    ld_bin_bp = 1000
    ld_scale_bp = 60_000.0
    ld_mixture = 0.30
    ld_rho = math.exp(-ld_bin_bp / ld_scale_bp)

    for chromosome, info in CHROMOSOMES.items():
        marker_count = info["markers"]
        chromosome_length = info["length_bp"]
        positions = (
            (np.arange(marker_count, dtype=np.float64) + 0.5)
            * chromosome_length
            / marker_count
        ).astype(np.int32)
        maf = maf_by_chromosome[chromosome]

        if scenario == "context_calibrated":
            segment_count = int(math.ceil(chromosome_length / ld_bin_bp))
            burn_in = 1000
            white_noise = rng.standard_normal(
                (N_SAMPLES, segment_count + burn_in), dtype=np.float32
            )
            factor_scale = math.sqrt(1 - ld_rho * ld_rho)
            factors = lfilter(
                [factor_scale], [1.0, -ld_rho], white_noise, axis=1
            ).astype(np.float32, copy=False)[:, burn_in:]
            del white_noise
        else:
            factors = None

        starts = np.linspace(0, marker_count, info["blocks"] + 1, dtype=int)
        for block_number, (start, stop) in enumerate(zip(starts[:-1], starts[1:]), 1):
            block_maf = maf[start:stop]
            width = stop - start
            if scenario == "blind":
                random_values = rng.random((N_SAMPLES, width), dtype=np.float32)
                block_genotypes = (random_values < block_maf[None, :]).T.astype(
                    np.uint8, copy=False
                )
                del random_values
            else:
                segment_ids = positions[start:stop] // ld_bin_bp
                latent = rng.standard_normal(
                    (N_SAMPLES, width), dtype=np.float32
                )
                latent *= math.sqrt(1 - ld_mixture)
                latent += factors[:, segment_ids] * math.sqrt(ld_mixture)
                thresholds = ndtri(block_maf).astype(np.float32)
                block_genotypes = (latent < thresholds[None, :]).T.astype(
                    np.uint8, copy=False
                )
                del latent

            block_path = genotype_dir / f"chr{chromosome}_block{block_number:03d}.npy"
            np.save(block_path, block_genotypes, allow_pickle=False)
            block_record = {
                "chromosome": chromosome,
                "block": block_number,
                "start_index": int(start),
                "stop_index": int(stop),
                "global_start": int(global_offset + start),
                "path": str(block_path.relative_to(out_dir)),
            }
            blocks.append(block_record)

            local_causal = [
                (global_id - global_offset, causal_index)
                for global_id, causal_index in global_to_causal.items()
                if global_offset + start <= global_id < global_offset + stop
            ]
            for chromosome_index, causal_index in local_causal:
                local_index = chromosome_index - start
                causal_dosages[:, causal_index] = block_genotypes[local_index]
                causal_mafs[causal_index] = maf[chromosome_index]
                causal_observed[causal_index] = True

            print(
                f"Generated {scenario}: chr{chromosome} block {block_number:03d}/"
                f"{info['blocks']} ({width:,} SNPs)",
                flush=True,
            )
        global_offset += marker_count
        del factors

    if not causal_observed.all() or not np.isfinite(causal_dosages).all():
        raise RuntimeError("Failed to capture all planted causal-marker genotypes.")
    if not np.isfinite(causal_mafs).all() or np.any(causal_mafs <= 0) or np.any(causal_mafs >= 1):
        raise RuntimeError("Invalid simulated causal-marker allele frequencies.")

    effects = rng.normal(size=N_CAUSAL)
    effects /= np.linalg.norm(effects)
    standardized_causal = (
        causal_dosages - causal_mafs[None, :]
    ) / np.sqrt(causal_mafs[None, :] * (1 - causal_mafs[None, :]))
    raw_genetic_score = np.sum(
        standardized_causal * effects[None, :], axis=1, dtype=np.float64
    )
    if not np.isfinite(raw_genetic_score).all() or raw_genetic_score.std() == 0:
        raise RuntimeError("Generated causal effects did not produce a finite phenotype signal.")
    genetic_signal = raw_genetic_score - raw_genetic_score.mean()
    genetic_signal /= genetic_signal.std(ddof=1)
    genetic_signal *= math.sqrt(TARGET_LATENT_SIGNAL_FRACTION)
    noise = rng.normal(
        loc=0.0,
        scale=math.sqrt(1 - TARGET_LATENT_SIGNAL_FRACTION),
        size=N_SAMPLES,
    )
    latent_phenotype = genetic_signal + noise
    latent_phenotype = (
        latent_phenotype - latent_phenotype.mean()
    ) / latent_phenotype.std(ddof=1)
    if not np.isfinite(latent_phenotype).all():
        raise RuntimeError("Synthetic phenotype contains non-finite values.")

    if scenario == "context_calibrated":
        assert phenotype_reference is not None
        probabilities = np.clip(norm.cdf(latent_phenotype), 1e-6, 1 - 1e-6)
        synthetic_phenotype = np.quantile(phenotype_reference, probabilities)
    else:
        synthetic_phenotype = latent_phenotype

    pd.DataFrame(
        {"Taxa": sample_ids, "FT10_simulated": synthetic_phenotype}
    ).to_csv(out_dir / "phenotypes.csv", index=False)

    causal_records = []
    for global_id, maf_value, effect in zip(causal_global_ids, causal_mafs, effects):
        for chromosome, offset in offsets.items():
            count = CHROMOSOMES[chromosome]["markers"]
            if offset <= global_id < offset + count:
                chromosome_index = global_id - offset
                causal_records.append(
                    {
                        "Marker": marker_label(chromosome, chromosome_index),
                        "Chr": chromosome,
                        "Pos": marker_position(chromosome, chromosome_index),
                        "MAF": float(maf_value),
                        "standardized_effect_weight": float(effect),
                    }
                )
                break
    pd.DataFrame(causal_records).sort_values(["Chr", "Pos"]).to_csv(
        out_dir / "causal_truth_POSTHOC_ONLY.csv", index=False
    )

    context_description = (
        {
            "maf_generation": "Empirical MAF-bin frequencies from all 3,557,529 filtered markers; MAF sampled uniformly within the observed bins",
            "ld_generation": "1-kb latent Gaussian segments with AR(1) correlation at 60-kb scale and 0.30 shared-factor variance",
            "phenotype_marginal": "Fixed empirical FT10 quantile map from the source phenotype values; no sample IDs or genotype-phenotype pairs reused",
            "phenotype_calibration_file": str(
                SOURCE_DIR / "ft10_gapit_phenotype.csv"
            ),
        }
        if scenario == "context_calibrated"
        else {
            "maf_generation": "Independent Uniform(0.05, 0.50) marker frequencies",
            "ld_generation": "Markers and accessions generated independently; no linkage",
            "phenotype_marginal": "Standardized latent quantitative phenotype; no empirical FT10 values used",
        }
    )
    manifest = {
        "scenario": scenario,
        "random_seed": seed,
        "n_samples": N_SAMPLES,
        "n_markers": N_MARKERS,
        "chromosomes": CHROMOSOMES,
        "n_causal_markers": N_CAUSAL,
        "target_latent_signal_variance_fraction": TARGET_LATENT_SIGNAL_FRACTION,
        "context": context_description,
        "blocks": blocks,
    }
    write_json(out_dir / "simulation_manifest.json", manifest)
    write_json(
        out_dir / "calibration_summary.json",
        {
            "n_samples": N_SAMPLES,
            "n_markers": N_MARKERS,
            "chromosomes": CHROMOSOMES,
            "maf_bin_edges": MAF_BINS if scenario == "context_calibrated" else None,
            "maf_bin_counts": MAF_COUNTS if scenario == "context_calibrated" else None,
            "ld_sample_median_r2": (
                LD_DISTANCE_SUMMARY if scenario == "context_calibrated" else None
            ),
            "ft10_reference_quantiles": (
                {
                    str(p): float(np.quantile(phenotype_reference, p))
                    for p in [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]
                }
                if phenotype_reference is not None
                else None
            ),
            "post_generation_phenotype_mean": float(np.mean(synthetic_phenotype)),
            "post_generation_phenotype_sd": float(
                np.std(synthetic_phenotype, ddof=1)
            ),
            "post_generation_phenotype_min": float(np.min(synthetic_phenotype)),
            "post_generation_phenotype_max": float(np.max(synthetic_phenotype)),
        },
    )
    print(f"Generated {N_MARKERS:,} synthetic markers in {len(blocks)} blocks.")


def make_models(
    seed: int,
    *,
    rf_trees: int = STAGE1_RF_TREES,
    gb_trees: int = STAGE1_GB_TREES,
) -> tuple[RandomForestRegressor, GradientBoostingRegressor]:
    rf = RandomForestRegressor(
        n_estimators=rf_trees,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed,
    )
    gb = GradientBoostingRegressor(
        n_estimators=gb_trees,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        max_features="sqrt",
        random_state=seed,
    )
    return rf, gb


def marker_details(global_ids: np.ndarray) -> pd.DataFrame:
    chromosome_ids = np.empty(len(global_ids), dtype=np.int8)
    chromosome_indices = np.empty(len(global_ids), dtype=np.int64)
    offsets = {}
    offset = 0
    for chromosome, info in CHROMOSOMES.items():
        offsets[chromosome] = offset
        offset += info["markers"]
    for idx, global_id in enumerate(global_ids):
        for chromosome, info in CHROMOSOMES.items():
            start = offsets[chromosome]
            if start <= global_id < start + info["markers"]:
                chromosome_ids[idx] = chromosome
                chromosome_indices[idx] = global_id - start
                break
    positions = [
        marker_position(int(chromosome), int(chrom_index))
        for chromosome, chrom_index in zip(chromosome_ids, chromosome_indices)
    ]
    return pd.DataFrame(
        {
            "global_index": global_ids,
            "Marker": [
                marker_label(int(chromosome), int(chrom_index))
                for chromosome, chrom_index in zip(chromosome_ids, chromosome_indices)
            ],
            "Chr": chromosome_ids,
            "Pos": positions,
        }
    )


def analyze_dataset(
    out_dir: Path,
    *,
    seed_override: int | None = None,
    outer_folds_override: int | None = None,
    stage1_rf_trees: int = STAGE1_RF_TREES,
    stage1_gb_trees: int = STAGE1_GB_TREES,
    stage2_rf_trees: int = STAGE2_RF_TREES,
    stage2_gb_trees: int = STAGE2_GB_TREES,
) -> None:
    base_seed = seed_override if seed_override is not None else BASE_SEED
    outer_folds = outer_folds_override if outer_folds_override is not None else OUTER_FOLDS
    manifest = json.loads((out_dir / "simulation_manifest.json").read_text())
    phenotype = pd.read_csv(out_dir / "phenotypes.csv")
    y = phenotype["FT10_simulated"].to_numpy(dtype=float)
    blocks = manifest["blocks"]
    folds = KFold(n_splits=outer_folds, shuffle=True, random_state=base_seed)
    rf_oof = np.full(len(y), np.nan)
    gb_oof = np.full(len(y), np.nan)
    fold_rows: list[dict[str, float | int]] = []
    block_screen_rows: list[dict[str, int | float]] = []
    rf_importance_sum: dict[int, float] = {}
    gb_importance_sum: dict[int, float] = {}
    rf_selected_count: dict[int, int] = {}
    gb_selected_count: dict[int, int] = {}

    for fold_number, (train_indices, test_indices) in enumerate(folds.split(y), 1):
        print(
            f"\n{manifest['scenario']}: starting outer fold {fold_number}/{outer_folds}",
            flush=True,
        )
        rf_candidates: list[int] = []
        gb_candidates: list[int] = []

        for block in blocks:
            path = out_dir / block["path"]
            genotypes = np.load(path, mmap_mode="r", allow_pickle=False)
            train_x = np.asarray(genotypes[:, train_indices].T, dtype=np.uint8)
            rf, gb = make_models(
                base_seed
                + 100_000 * fold_number
                + 1000 * int(block["chromosome"])
                + int(block["block"]),
                rf_trees=stage1_rf_trees,
                gb_trees=stage1_gb_trees,
            )
            rf.fit(train_x, y[train_indices])
            gb.fit(train_x, y[train_indices])

            rf_count = min(TOP_K_PER_BLOCK_PER_MODEL, train_x.shape[1])
            gb_count = rf_count
            rf_top = np.argpartition(rf.feature_importances_, -rf_count)[-rf_count:]
            gb_top = np.argpartition(gb.feature_importances_, -gb_count)[-gb_count:]
            global_start = int(block["global_start"])
            rf_ids = global_start + rf_top
            gb_ids = global_start + gb_top
            rf_candidates.extend(rf_ids.tolist())
            gb_candidates.extend(gb_ids.tolist())
            block_screen_rows.append(
                {
                    "fold": fold_number,
                    "chromosome": int(block["chromosome"]),
                    "block": int(block["block"]),
                    "markers_screened": int(train_x.shape[1]),
                    "rf_selected": int(len(rf_ids)),
                    "gb_selected": int(len(gb_ids)),
                }
            )
            print(
                f"  screened chr{block['chromosome']} block {block['block']:03d}",
                flush=True,
            )
            del train_x, genotypes, rf, gb

        candidates = np.asarray(
            sorted(set(rf_candidates).union(gb_candidates)), dtype=np.int64
        )
        rf_selected_set = set(rf_candidates)
        gb_selected_set = set(gb_candidates)
        for marker_id in rf_selected_set:
            rf_selected_count[marker_id] = rf_selected_count.get(marker_id, 0) + 1
        for marker_id in gb_selected_set:
            gb_selected_count[marker_id] = gb_selected_count.get(marker_id, 0) + 1

        x_train = np.empty((len(train_indices), len(candidates)), dtype=np.uint8)
        x_test = np.empty((len(test_indices), len(candidates)), dtype=np.uint8)
        for block in blocks:
            global_start = int(block["global_start"])
            block_width = int(block["stop_index"]) - int(block["start_index"])
            left = np.searchsorted(candidates, global_start)
            right = np.searchsorted(candidates, global_start + block_width)
            if left == right:
                continue
            chosen_global = candidates[left:right]
            local_rows = chosen_global - global_start
            genotypes = np.load(out_dir / block["path"], mmap_mode="r", allow_pickle=False)
            values = genotypes[local_rows, :]
            x_train[:, left:right] = values[:, train_indices].T
            x_test[:, left:right] = values[:, test_indices].T
            del values, genotypes

        rf_final = RandomForestRegressor(
            n_estimators=stage2_rf_trees,
            max_features="sqrt",
            n_jobs=-1,
            random_state=base_seed + 500_000 + fold_number,
        )
        gb_final = GradientBoostingRegressor(
            n_estimators=stage2_gb_trees,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            max_features="sqrt",
            random_state=base_seed + 600_000 + fold_number,
        )
        rf_final.fit(x_train, y[train_indices])
        gb_final.fit(x_train, y[train_indices])
        rf_oof[test_indices] = rf_final.predict(x_test)
        gb_oof[test_indices] = gb_final.predict(x_test)

        for marker_id, importance in zip(candidates, rf_final.feature_importances_):
            rf_importance_sum[int(marker_id)] = (
                rf_importance_sum.get(int(marker_id), 0.0) + float(importance)
            )
        for marker_id, importance in zip(candidates, gb_final.feature_importances_):
            gb_importance_sum[int(marker_id)] = (
                gb_importance_sum.get(int(marker_id), 0.0) + float(importance)
            )

        fold_rows.append(
            {
                "fold": fold_number,
                "test_n": int(len(test_indices)),
                "finalist_count": int(len(candidates)),
                "rf_r2": float(r2_score(y[test_indices], rf_oof[test_indices])),
                "gb_r2": float(r2_score(y[test_indices], gb_oof[test_indices])),
                "rf_rmse": float(
                    math.sqrt(mean_squared_error(y[test_indices], rf_oof[test_indices]))
                ),
                "gb_rmse": float(
                    math.sqrt(mean_squared_error(y[test_indices], gb_oof[test_indices]))
                ),
            }
        )
        print(
            f"  fold {fold_number}: finalists={len(candidates):,}; "
            f"RF R2={fold_rows[-1]['rf_r2']:.3f}; GB R2={fold_rows[-1]['gb_r2']:.3f}",
            flush=True,
        )
        del x_train, x_test, rf_final, gb_final

    if np.isnan(rf_oof).any() or np.isnan(gb_oof).any():
        raise RuntimeError("Some accessions did not receive out-of-fold predictions.")

    pd.DataFrame(fold_rows).to_csv(out_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(block_screen_rows).to_csv(
        out_dir / "per_fold_block_screening.csv", index=False
    )
    pd.DataFrame(
        {
            "Taxa": phenotype["Taxa"],
            "FT10_simulated": y,
            "RF_oof_prediction": rf_oof,
            "GB_oof_prediction": gb_oof,
        }
    ).to_csv(out_dir / "out_of_fold_predictions.csv", index=False)

    all_importance_ids = np.asarray(
        sorted(set(rf_importance_sum).union(gb_importance_sum)), dtype=np.int64
    )
    details = marker_details(all_importance_ids)
    result_table = details.copy()
    result_table["RF_mean_importance"] = [
        rf_importance_sum.get(int(marker_id), 0.0) / outer_folds
        for marker_id in all_importance_ids
    ]
    result_table["GB_mean_importance"] = [
        gb_importance_sum.get(int(marker_id), 0.0) / outer_folds
        for marker_id in all_importance_ids
    ]
    result_table["RF_selection_folds"] = [
        rf_selected_count.get(int(marker_id), 0) for marker_id in all_importance_ids
    ]
    result_table["GB_selection_folds"] = [
        gb_selected_count.get(int(marker_id), 0) for marker_id in all_importance_ids
    ]
    result_table["RF_rank"] = result_table["RF_mean_importance"].rank(
        ascending=False, method="min"
    )
    result_table["GB_rank"] = result_table["GB_mean_importance"].rank(
        ascending=False, method="min"
    )
    result_table.sort_values("RF_mean_importance", ascending=False).to_csv(
        out_dir / "screened_candidates_and_importance.csv", index=False
    )
    top_rf = result_table.nlargest(50, "RF_mean_importance").copy()
    top_gb = result_table.nlargest(50, "GB_mean_importance").copy()
    top_rf.to_csv(out_dir / "top50_random_forest.csv", index=False)
    top_gb.to_csv(out_dir / "top50_gradient_boosting.csv", index=False)

    rf_overlap = len(set(top_rf["Marker"]).intersection(top_gb["Marker"]))
    summary = {
        "scenario": manifest["scenario"],
        "samples": int(len(y)),
        "markers_screened_in_each_fold": N_MARKERS,
        "outer_folds": outer_folds,
        "rf_oof_r2": float(r2_score(y, rf_oof)),
        "gb_oof_r2": float(r2_score(y, gb_oof)),
        "rf_oof_rmse": float(math.sqrt(mean_squared_error(y, rf_oof))),
        "gb_oof_rmse": float(math.sqrt(mean_squared_error(y, gb_oof))),
        "rf_oof_mae": float(mean_absolute_error(y, rf_oof)),
        "gb_oof_mae": float(mean_absolute_error(y, gb_oof)),
        "top50_overlap": rf_overlap,
        "candidates_in_importance_table": int(len(result_table)),
        "method": "Fold-specific per-block screening followed by joint finalist-model fitting; screening and fitting use outer-training samples only.",
    }
    write_json(out_dir / "analysis_summary.json", summary)

    # Causal truth is loaded only after all cross-validated predictions,
    # importance ranks, and top-marker files have been written.
    truth = pd.read_csv(out_dir / "causal_truth_POSTHOC_ONLY.csv")
    recovery_rows = []
    for model, top, rank_col in [
        ("Random Forest", top_rf, "RF_rank"),
        ("Gradient Boosting", top_gb, "GB_rank"),
    ]:
        model_ranks = result_table.set_index("Marker")[rank_col].to_dict()
        model_selected = result_table.set_index("Marker")[
            "RF_selection_folds" if model == "Random Forest" else "GB_selection_folds"
        ].to_dict()
        selected_top50 = set(top["Marker"])
        for _, causal in truth.iterrows():
            nearby = result_table[
                (result_table["Chr"] == int(causal["Chr"]))
                & ((result_table["Pos"] - int(causal["Pos"])).abs() <= 60_000)
            ]
            recovery_rows.append(
                {
                    "model": model,
                    "causal_marker": causal["Marker"],
                    "Chr": int(causal["Chr"]),
                    "Pos": int(causal["Pos"]),
                    "final_importance_rank": model_ranks.get(causal["Marker"]),
                    "selected_in_outer_training_folds": model_selected.get(
                        causal["Marker"], 0
                    ),
                    "exactly_in_top50": causal["Marker"] in selected_top50,
                    "any_top50_marker_within_60kb": bool(
                        np.any((top["Chr"] == int(causal["Chr"]))
                               & ((top["Pos"] - int(causal["Pos"])).abs() <= 60_000))
                    ),
                    "best_top50_distance_bp": (
                        int((top.loc[top["Chr"] == int(causal["Chr"]), "Pos"]
                             - int(causal["Pos"])).abs().min())
                        if np.any(top["Chr"] == int(causal["Chr"]))
                        else None
                    ),
                }
            )
    pd.DataFrame(recovery_rows).to_csv(out_dir / "causal_recovery_posthoc.csv", index=False)

    with (out_dir / "run_summary.txt").open("w") as handle:
        handle.write(f"Scenario: {manifest['scenario']}\n")
        handle.write(f"Samples: {len(y)}\n")
        handle.write(f"Markers screened in each fold: {N_MARKERS:,}\n")
        handle.write(f"Outer folds: {outer_folds}; marker selection repeated inside each fold\n")
        handle.write(f"RF out-of-fold R2: {summary['rf_oof_r2']:.4f}\n")
        handle.write(f"GB out-of-fold R2: {summary['gb_oof_r2']:.4f}\n")
        handle.write(f"RF out-of-fold RMSE: {summary['rf_oof_rmse']:.4f}\n")
        handle.write(f"GB out-of-fold RMSE: {summary['gb_oof_rmse']:.4f}\n")
        handle.write(f"Top-50 overlap: {rf_overlap}/50\n")
        handle.write("Causal truth is used only in the post-hoc recovery file.\n")
    print(f"\nCompleted {manifest['scenario']}: {json.dumps(summary, indent=2)}")


def write_prior_locus_comparison(out_dir: Path) -> None:
    regions = {
        "FT": {
            "chromosome": 1,
            "lower_bp": 24_279_560,
            "upper_bp": 24_399_560,
            "center_bp": 24_339_560,
        },
        "DOG1": {
            "chromosome": 5,
            "lower_bp": 18_530_000,
            "upper_bp": 18_650_000,
            "center_bp": 18_590_000,
        },
        "ZTL": {
            "chromosome": 5,
            "lower_bp": 23_100_000,
            "upper_bp": 23_300_000,
            "center_bp": 23_200_000,
        },
    }
    records = []
    for region, bounds in regions.items():
        chromosome = bounds["chromosome"]
        lower = bounds["lower_bp"]
        upper = bounds["upper_bp"]
        center = bounds["center_bp"]
        planted = pd.read_csv(out_dir / "causal_truth_POSTHOC_ONLY.csv")
        planted = planted[
            (planted["Chr"] == chromosome)
            & planted["Pos"].between(lower, upper)
        ]
        for model, filename, importance_column in [
            ("Random Forest", "top50_random_forest.csv", "RF_mean_importance"),
            ("Gradient Boosting", "top50_gradient_boosting.csv", "GB_mean_importance"),
        ]:
            top = pd.read_csv(out_dir / filename)
            chromosome_top = top[top["Chr"] == chromosome].copy()
            chromosome_top["distance_to_region_bp"] = np.where(
                chromosome_top["Pos"] < lower,
                lower - chromosome_top["Pos"],
                np.where(chromosome_top["Pos"] > upper, chromosome_top["Pos"] - upper, 0),
            )
            selected = chromosome_top[chromosome_top["distance_to_region_bp"] == 0]
            closest = chromosome_top.sort_values("distance_to_region_bp")
            records.append(
                {
                    "model": model,
                    "prior_region": region,
                    "comparison_window": f"{lower}-{upper}",
                    "planted_causal_markers_in_window": int(len(planted)),
                    "top50_markers_in_window": int(len(selected)),
                    "closest_top50_marker": (
                        closest.iloc[0]["Marker"] if not closest.empty else ""
                    ),
                    "closest_distance_to_region_bp": (
                        int(closest.iloc[0]["distance_to_region_bp"])
                        if not closest.empty
                        else ""
                    ),
                    "highest_importance_marker": (
                        selected.sort_values(importance_column, ascending=False)
                        .iloc[0]["Marker"]
                        if not selected.empty
                        else ""
                    ),
                }
            )
    pd.DataFrame(records).to_csv(
        out_dir / "prior_real_loci_comparison_POSTHOC_ONLY.csv", index=False
    )


def main() -> None:
    global SOURCE_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        required=True,
        choices=["blind", "context_calibrated"],
        help="Which fully synthetic data scenario to generate and analyze.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=SOURCE_DIR,
        help="Source folder used only for context calibration.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Analyze already-generated synthetic inputs without regenerating them.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate the synthetic dataset without fitting any ML models.",
    )
    parser.add_argument(
        "--posthoc-only",
        action="store_true",
        help="Compare finished rankings to previously reported loci, after analysis.",
    )
    args = parser.parse_args()
    SOURCE_DIR = args.source_dir
    out_dir = ROOT / (
        "blind_simulation"
        if args.scenario == "blind"
        else "context_calibrated_simulation"
    )
    if args.posthoc_only:
        write_prior_locus_comparison(out_dir)
    elif not args.analyze_only:
        generate_genotypes_and_phenotypes(args.scenario, out_dir)
        if not args.generate_only:
            analyze_dataset(out_dir)
    elif not args.generate_only:
        analyze_dataset(out_dir)


if __name__ == "__main__":
    main()
