#!/usr/bin/env python3
"""Genome-wide, nested ML benchmark on 1000 Genomes EUR genotypes.

The response is the exact simulated phenotype used in the preceding pilot.
The target/causal marker identities are withheld from all fitting and only
loaded after model predictions and rankings have been saved.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression


RUN_DIR = Path(__file__).resolve().parent
PILOT_DIR = RUN_DIR.parent / "05_human_pilot"
DATA_DIR = RUN_DIR / "genotype_chunks"
VCF_BASE = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
    "1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/"
    "1kGP_high_coverage_Illumina.chr{chrom}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
)
N_EUR = 503
MAF_MIN = 0.05
CHUNK_VARIANTS = 25_000
SCAN_WINDOW_BP = 5_000_000
PC_SPACING_BP = 100_000
N_PCS = 10
N_OUTER_FOLDS = 5
TOP_PER_CHR_PER_MODEL = 10
TOP_REPORT = 50
BASE_SEED = 20260929
CHROMS = list(range(1, 23))
EXTRACTION_WORKERS = 4
SCREEN_RF_TREES = 300
SCREEN_GB_TREES = 150


def load_samples() -> list[str]:
    panel = pd.read_csv(PILOT_DIR / "1000genomes_sample_panel.tsv", sep="\t")
    samples = panel.loc[panel["super_pop"] == "EUR", "sample"].tolist()
    if len(samples) != N_EUR or len(set(samples)) != N_EUR:
        raise RuntimeError(f"Expected {N_EUR} unique EUR sample IDs; found {len(samples)}.")
    return samples


def save_chunk(chrom: int, chunk_no: int, rows: list[np.ndarray],
               metadata: list[dict]) -> dict:
    stem = f"chr{chrom:02d}_chunk{chunk_no:04d}"
    matrix_path = DATA_DIR / f"{stem}.npy"
    map_path = DATA_DIR / f"{stem}.tsv.gz"
    np.save(matrix_path, np.vstack(rows).astype(np.int8, copy=False))
    pd.DataFrame(metadata).to_csv(map_path, sep="\t", index=False, compression="gzip")
    return {
        "chrom": chrom,
        "chunk_no": chunk_no,
        "n_markers": len(rows),
        "genotype_file": matrix_path.name,
        "marker_file": map_path.name,
    }


def parse_record(record, chrom: int):
    if len(record.ref) != 1 or record.ref not in "ACGT":
        return None
    alts = record.alts
    if alts is None or len(alts) != 1 or len(alts[0]) != 1 or alts[0] not in "ACGT":
        return None

    annotated_maf = record.info.get("MAF_EUR_unrel")
    if annotated_maf is None or len(annotated_maf) != 1:
        raise RuntimeError(
            f"Missing single-alt MAF_EUR_unrel annotation at chr{chrom}:{record.pos}."
        )
    annotated_maf = float(annotated_maf[0])
    if not np.isfinite(annotated_maf):
        raise RuntimeError(f"Non-finite MAF_EUR_unrel at chr{chrom}:{record.pos}.")
    if annotated_maf + 1e-6 < MAF_MIN:
        return None

    dosage = np.empty(N_EUR, dtype=np.int8)
    for i, call in enumerate(record.samples.values()):
        gt = call.get("GT")
        if gt is None or len(gt) != 2 or gt[0] not in (0, 1) or gt[1] not in (0, 1):
            return None
        dosage[i] = gt[0] + gt[1]

    alt_frequency = float(dosage.sum()) / (2.0 * N_EUR)
    maf = min(alt_frequency, 1.0 - alt_frequency)
    if abs(maf - annotated_maf) > 1e-6:
        raise RuntimeError(
            f"VCF EUR-unrelated MAF disagrees with selected-sample MAF at "
            f"chr{chrom}:{record.pos} ({annotated_maf} vs {maf})."
        )
    if maf < MAF_MIN:
        return None

    metadata = {
        "Chr": chrom,
        "Pos": int(record.pos),
        "Ref": record.ref,
        "Alt": alts[0],
        "MAF": maf,
        "rsid": record.id or ".",
    }
    return dosage, metadata


def extract_chromosome(chrom: int, samples: list[str]) -> dict:
    complete = DATA_DIR / f"chr{chrom:02d}.complete.json"
    if complete.exists():
        return json.loads(complete.read_text())

    for partial in DATA_DIR.glob(f"chr{chrom:02d}_chunk*"):
        if partial.is_file():
            partial.unlink()
    pc_path = DATA_DIR / f"chr{chrom:02d}_pc.npy"
    if pc_path.exists():
        pc_path.unlink()

    url = VCF_BASE.format(chrom=chrom)
    probe = pysam.VariantFile(url)
    absent = sorted(set(samples) - set(probe.header.samples))
    if absent:
        raise RuntimeError(f"EUR samples missing from chr{chrom} VCF: {absent[:5]}")
    contig = f"chr{chrom}"
    if contig not in probe.header.contigs or probe.header.contigs[contig].length is None:
        raise RuntimeError(f"Missing chromosome length for {contig} in the VCF header.")
    contig_length = int(probe.header.contigs[contig].length)
    probe.close()
    rows: list[np.ndarray] = []
    metadata: list[dict] = []
    chrom_pc_rows: list[np.ndarray] = []
    chrom_chunks: list[dict] = []
    chunk_no = 0
    n_seen = 0
    next_pc_pos = 1
    started = time.time()

    for start0 in range(0, contig_length, SCAN_WINDOW_BP):
        end0 = min(start0 + SCAN_WINDOW_BP, contig_length)
        variant_file = pysam.VariantFile(url)
        variant_file.subset_samples(samples)
        try:
            for record in variant_file.fetch(contig, start0, end0):
                n_seen += 1
                parsed = parse_record(record, chrom)
                if parsed is None:
                    continue
                dosage, marker = parsed
                if marker["Pos"] >= next_pc_pos:
                    chrom_pc_rows.append(dosage.copy())
                    next_pc_pos = marker["Pos"] + PC_SPACING_BP
                rows.append(dosage)
                metadata.append(marker)
                if len(rows) >= CHUNK_VARIANTS:
                    chrom_chunks.append(save_chunk(chrom, chunk_no, rows, metadata))
                    chunk_no += 1
                    rows, metadata = [], []
        finally:
            variant_file.close()
        print(
            f"chr{chrom}: scanned through {end0:,}/{contig_length:,} bp; "
            f"{sum(c['n_markers'] for c in chrom_chunks) + len(rows):,} common SNVs retained",
            flush=True,
        )

    if rows:
        chrom_chunks.append(save_chunk(chrom, chunk_no, rows, metadata))
    if not chrom_chunks:
        raise RuntimeError(f"No common biallelic SNVs passed QC on chromosome {chrom}.")
    if not chrom_pc_rows:
        raise RuntimeError(f"No PCA markers retained on chromosome {chrom}.")
    np.save(pc_path, np.vstack(chrom_pc_rows).astype(np.int8, copy=False))
    chromosome_manifest = {
        "chrom": chrom,
        "n_markers": sum(c["n_markers"] for c in chrom_chunks),
        "n_vcf_records_scanned": n_seen,
        "chunks": chrom_chunks,
        "elapsed_seconds": time.time() - started,
    }
    complete.write_text(json.dumps(chromosome_manifest, indent=2))
    print(
        f"chr{chrom}: completed {n_seen:,} records; "
        f"{chromosome_manifest['n_markers']:,} variants passed QC in "
        f"{chromosome_manifest['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return chromosome_manifest


def extract_genome(samples: list[str]) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RUN_DIR / "extraction_manifest.json"
    done_files = [DATA_DIR / f"chr{c:02d}.complete.json" for c in CHROMS]
    if all(path.exists() for path in done_files) and manifest_path.exists():
        return json.loads(manifest_path.read_text())

    start_all = time.time()
    unfinished = [chrom for chrom in CHROMS
                  if not (DATA_DIR / f"chr{chrom:02d}.complete.json").exists()]
    with ThreadPoolExecutor(max_workers=EXTRACTION_WORKERS) as pool:
        futures = {pool.submit(extract_chromosome, chrom, samples): chrom
                   for chrom in unfinished}
        for future in as_completed(futures):
            future.result()

    chromosome_manifests = [
        json.loads((DATA_DIR / f"chr{chrom:02d}.complete.json").read_text())
        for chrom in CHROMS
    ]
    chunks = [chunk for item in chromosome_manifests for chunk in item["chunks"]]
    total_markers = sum(item["n_markers"] for item in chromosome_manifests)
    pc_rows = [
        dosage
        for chrom in CHROMS
        for dosage in np.load(DATA_DIR / f"chr{chrom:02d}_pc.npy")
    ]

    if not pc_rows:
        raise RuntimeError("No common variants available for ancestry PCA.")
    pc_matrix = np.vstack(pc_rows).astype(np.float64).T
    means = pc_matrix.mean(axis=0)
    sds = pc_matrix.std(axis=0, ddof=1)
    variable = sds > 0
    pc_matrix = pc_matrix[:, variable]
    pc_matrix = (pc_matrix - means[variable]) / sds[variable]
    n_components = min(N_PCS, N_EUR - 1, pc_matrix.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=BASE_SEED)
    pcs = pca.fit_transform(pc_matrix)
    pc_frame = pd.DataFrame(pcs, columns=[f"PC{i + 1}" for i in range(n_components)])
    pc_frame.insert(0, "sample", samples)
    pc_frame.to_csv(RUN_DIR / "ancestry_pcs.csv", index=False)
    pd.DataFrame({
        "PC": [f"PC{i + 1}" for i in range(n_components)],
        "variance_explained": pca.explained_variance_ratio_,
    }).to_csv(RUN_DIR / "ancestry_pc_variance.csv", index=False)

    manifest = {
        "source": "1000 Genomes Project 30x high-coverage phased panel, GRCh38",
        "vcf_base_url": VCF_BASE,
        "sample_panel": str(PILOT_DIR / "1000genomes_sample_panel.tsv"),
        "samples": samples,
        "n_samples": N_EUR,
        "chromosomes": CHROMS,
        "extraction_workers": EXTRACTION_WORKERS,
        "variant_filter": (
            f"Autosomal biallelic SNVs, called diploid genotypes in all {N_EUR} EUR samples, "
            f"sample MAF >= {MAF_MIN}; VCF MAF_EUR_unrel used as a conservative "
            f"prefilter and confirmed against the selected-sample genotypes"
        ),
        "n_markers": int(total_markers),
        "chunk_variants": CHUNK_VARIANTS,
        "scan_window_bp": SCAN_WINDOW_BP,
        "n_pca_markers": int(len(pc_rows)),
        "pca_marker_spacing_bp": PC_SPACING_BP,
        "n_pcs": int(n_components),
        "chunks": chunks,
        "elapsed_seconds": time.time() - start_all,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Extracted {total_markers:,} autosomal common SNVs.", flush=True)
    return manifest


def read_chunks(manifest: dict, chrom: int | None = None):
    for chunk in manifest["chunks"]:
        if chrom is None or int(chunk["chrom"]) == chrom:
            matrix = np.load(DATA_DIR / chunk["genotype_file"], mmap_mode="r")
            markers = pd.read_csv(DATA_DIR / chunk["marker_file"], sep="\t")
            yield chunk, matrix, markers


def regression_scan(genotypes: np.ndarray, q_cov: np.ndarray,
                    y_residual: np.ndarray, yss: float, df: int):
    """PC-adjusted additive regression statistics for marker rows."""
    g = np.asarray(genotypes, dtype=np.float64)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        projected = g @ q_cov
        cross = g @ y_residual
    if not np.isfinite(projected).all() or not np.isfinite(cross).all():
        raise FloatingPointError("Non-finite cross-products in the genome-wide association scan.")
    gss = (
        np.einsum("ij,ij->i", g, g, optimize=True)
        - np.einsum("ij,ij->i", projected, projected, optimize=True)
    )
    gss = np.maximum(gss, 1e-20)
    sse = np.maximum(yss - np.square(cross) / np.maximum(gss, 1e-20), 1e-20)
    t_stat = cross * np.sqrt(df / (np.maximum(gss, 1e-20) * sse))
    beta = cross / np.maximum(gss, 1e-20)
    pvalue = 2.0 * stats.t.sf(np.abs(t_stat), df)
    return beta, t_stat, pvalue


def make_covariates(pcs: np.ndarray, sample_idx: np.ndarray):
    c = np.column_stack([np.ones(len(sample_idx)), pcs[sample_idx]])
    q, _ = np.linalg.qr(c, mode="reduced")
    return q


def run_gwas(manifest: dict, y: np.ndarray, pcs: np.ndarray) -> pd.DataFrame:
    raw_path = RUN_DIR / "gwas_raw_intermediate.tsv.gz"
    pvalue_chunks: list[np.ndarray] = []
    top = pd.DataFrame()
    q_cov = make_covariates(pcs, np.arange(len(y)))
    y_resid = y - q_cov @ (q_cov.T @ y)
    yss = float(y_resid @ y_resid)
    df = len(y) - q_cov.shape[1] - 1
    if raw_path.exists():
        raw_path.unlink()

    total = 0
    with gzip.open(raw_path, "wt") as handle:
        header = True
        for _, matrix, markers in read_chunks(manifest):
            beta, t_stat, pvalue = regression_scan(matrix, q_cov, y_resid, yss, df)
            result = markers.copy()
            result["beta_PC_adjusted"] = beta
            result["t_PC_adjusted"] = t_stat
            result["p_PC_adjusted"] = pvalue
            result["global_idx"] = np.arange(total, total + len(result), dtype=np.int64)
            result.to_csv(handle, sep="\t", index=False, header=header)
            header = False
            pvalue_chunks.append(pvalue)
            top = pd.concat([top, result.nsmallest(1000, "p_PC_adjusted")], ignore_index=True)
            if len(top) > 2000:
                top = top.nsmallest(1000, "p_PC_adjusted").reset_index(drop=True)
            total += len(result)
            if total % 500_000 < CHUNK_VARIANTS:
                print(f"GWAS: {total:,}/{manifest['n_markers']:,} variants", flush=True)

    pvalues = np.concatenate(pvalue_chunks)
    if len(pvalues) != manifest["n_markers"]:
        raise RuntimeError("GWAS result count does not match the extraction manifest.")
    order = np.argsort(pvalues, kind="stable")
    ranks = np.arange(1, len(pvalues) + 1, dtype=np.float64)
    q_sorted = np.minimum.accumulate((pvalues[order] * len(pvalues) / ranks)[::-1])[::-1]
    qvalues = np.empty_like(q_sorted)
    qvalues[order] = np.minimum(q_sorted, 1.0)
    final_path = RUN_DIR / "genomewide_gwas_summary_stats.tsv.gz"
    offset = 0
    with gzip.open(final_path, "wt") as out:
        header = True
        for frame in pd.read_csv(raw_path, sep="\t", chunksize=100_000):
            frame["BH_FDR"] = qvalues[offset:offset + len(frame)]
            frame.drop(columns="global_idx").to_csv(
                out, sep="\t", index=False, header=header
            )
            header = False
            offset += len(frame)
    raw_path.unlink()

    top = top.sort_values("p_PC_adjusted").drop_duplicates(
        ["Chr", "Pos", "Ref", "Alt"]
    ).head(1000)
    top["BH_FDR"] = qvalues[top["global_idx"].to_numpy(dtype=np.int64)]
    top = top.drop(columns="global_idx")
    top.to_csv(RUN_DIR / "top1000_gwas_hits.tsv", sep="\t", index=False)
    return top


def adjust_fold(y_train: np.ndarray, pcs_train: np.ndarray):
    cov = np.column_stack([np.ones(len(y_train)), pcs_train])
    q, _ = np.linalg.qr(cov, mode="reduced")
    residual = y_train - q @ (q.T @ y_train)
    return q, residual, float(residual @ residual), len(y_train) - q.shape[1] - 1


def candidates_for_fold(manifest: dict, train_idx: np.ndarray,
                        y: np.ndarray, pcs: np.ndarray, fold: int) -> pd.DataFrame:
    q_cov, y_resid, _, _ = adjust_fold(y[train_idx], pcs[train_idx])
    candidates_by_model = {"rf": [], "gb": []}

    for chrom in CHROMS:
        chunk_rows = []
        marker_frames = []
        for chunk, matrix, markers in read_chunks(manifest, chrom=chrom):
            chunk_rows.append(np.asarray(matrix))
            current = markers.copy()
            current["chunk_file"] = chunk["genotype_file"]
            current["local_row"] = np.arange(len(current), dtype=np.int32)
            marker_frames.append(current)
        if not chunk_rows:
            raise RuntimeError(f"No extracted genotype chunks for chromosome {chrom}.")

        genotypes = np.concatenate(chunk_rows, axis=0)
        marker_map = pd.concat(marker_frames, ignore_index=True)
        x_train = np.ascontiguousarray(
            genotypes[:, train_idx].T, dtype=np.float32
        )
        rf = RandomForestRegressor(
            n_estimators=SCREEN_RF_TREES, max_features="sqrt",
            min_samples_leaf=2, n_jobs=4,
            random_state=BASE_SEED + 100_000 + fold * 100 + chrom,
        )
        gb = GradientBoostingRegressor(
            n_estimators=SCREEN_GB_TREES, max_depth=2, learning_rate=0.05,
            subsample=0.8, max_features="sqrt",
            random_state=BASE_SEED + 200_000 + fold * 100 + chrom,
        )
        for key, model in (("rf", rf), ("gb", gb)):
            model.fit(x_train, y_resid)
            top_indices = np.argsort(model.feature_importances_)[::-1][
                :TOP_PER_CHR_PER_MODEL
            ]
            selected = marker_map.iloc[top_indices].copy()
            selected["candidate_id"] = (
                "chr" + selected["Chr"].astype(str) + "_"
                + selected["Pos"].astype(str) + "_"
                + selected["Ref"].astype(str) + "_"
                + selected["Alt"].astype(str)
            )
            selected["stage1_importance"] = model.feature_importances_[top_indices]
            selected["selected_by"] = key
            candidates_by_model[key].append(selected)
        print(
            f"  fold {fold}: screened chr{chrom} "
            f"({genotypes.shape[0]:,} markers) with training-only RF/GB",
            flush=True,
        )
        del genotypes, chunk_rows, marker_frames, marker_map, x_train

    candidates = pd.concat(
        candidates_by_model["rf"] + candidates_by_model["gb"], ignore_index=True
    )
    candidates = candidates.drop_duplicates(
        ["candidate_id"], keep="first"
    ).reset_index(drop=True)
    candidates.to_csv(RUN_DIR / f"fold{fold}_training_candidates.tsv", sep="\t", index=False)
    return candidates


def candidate_matrix(candidates: pd.DataFrame, n_samples: int) -> np.ndarray:
    x = np.empty((n_samples, len(candidates)), dtype=np.float32)
    for chunk_name, group in candidates.groupby("chunk_file", sort=False):
        matrix = np.load(DATA_DIR / chunk_name, mmap_mode="r")
        for col, (_, row) in enumerate(group.iterrows()):
            x[:, int(row.name)] = matrix[int(row["local_row"]), :]
    return x


def make_ml_models(fold: int):
    rf = RandomForestRegressor(
        n_estimators=500, max_features="sqrt", min_samples_leaf=2,
        n_jobs=4, random_state=BASE_SEED + 50_000 + fold,
    )
    gb = GradientBoostingRegressor(
        n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8,
        max_features="sqrt", random_state=BASE_SEED + 60_000 + fold,
    )
    return rf, gb


def run_nested_ml(manifest: dict, y: np.ndarray, pcs: np.ndarray) -> dict:
    folds = KFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=BASE_SEED)
    predictions = {"rf": np.full(len(y), np.nan), "gb": np.full(len(y), np.nan)}
    pc_predictions = np.full(len(y), np.nan)
    fold_rows = []
    importance_sum = {"rf": {}, "gb": {}}
    selection_count = {"rf": {}, "gb": {}}

    for fold, (train_idx, test_idx) in enumerate(folds.split(y), 1):
        print(f"\n=== nested ML outer fold {fold}/{N_OUTER_FOLDS} ===", flush=True)
        candidates = candidates_for_fold(manifest, train_idx, y, pcs, fold)
        x_candidates = candidate_matrix(candidates, len(y))
        c_train = np.column_stack([np.ones(len(train_idx)), pcs[train_idx]])
        c_test = np.column_stack([np.ones(len(test_idx)), pcs[test_idx]])
        pc_model = LinearRegression(fit_intercept=False).fit(c_train, y[train_idx])
        pc_predictions[test_idx] = pc_model.predict(c_test)
        x = np.column_stack([pcs, x_candidates]).astype(np.float32, copy=False)

        rf, gb = make_ml_models(fold)
        for name, model in (("rf", rf), ("gb", gb)):
            model.fit(x[train_idx], y[train_idx])
            predictions[name][test_idx] = model.predict(x[test_idx])
            for i, marker_id in enumerate(candidates["candidate_id"]):
                importance_sum[name][marker_id] = (
                    importance_sum[name].get(marker_id, 0.0)
                    + float(model.feature_importances_[N_PCS + i]) / N_OUTER_FOLDS
                )
                selection_count[name][marker_id] = selection_count[name].get(marker_id, 0) + 1
        fold_rows.append({
            "fold": fold,
            "n_candidates": len(candidates),
            "rf_fold_r2": r2_score(y[test_idx], predictions["rf"][test_idx]),
            "gb_fold_r2": r2_score(y[test_idx], predictions["gb"][test_idx]),
            "pc_only_fold_r2": r2_score(y[test_idx], pc_predictions[test_idx]),
        })
        print(
            f"fold {fold}: {len(candidates)} candidates; "
            f"RF R2={fold_rows[-1]['rf_fold_r2']:.3f}, "
            f"GB R2={fold_rows[-1]['gb_fold_r2']:.3f}, "
            f"PC-only R2={fold_rows[-1]['pc_only_fold_r2']:.3f}",
            flush=True,
        )

    if any(np.isnan(v).any() for v in predictions.values()) or np.isnan(pc_predictions).any():
        raise RuntimeError("Some samples did not receive out-of-fold predictions.")
    pd.DataFrame(fold_rows).to_csv(RUN_DIR / "nested_cv_fold_metrics.csv", index=False)
    pd.DataFrame({
        "sample": load_samples(),
        "y_simulated": y,
        "rf_oof_prediction": predictions["rf"],
        "gb_oof_prediction": predictions["gb"],
        "pc_only_oof_prediction": pc_predictions,
    }).to_csv(RUN_DIR / "nested_cv_oof_predictions.csv", index=False)

    importance = pd.DataFrame({"Marker": sorted(set(importance_sum["rf"]) | set(importance_sum["gb"]))})
    for name in ("rf", "gb"):
        importance[f"{name.upper()}_mean_importance"] = [
            importance_sum[name].get(marker, 0.0) for marker in importance["Marker"]
        ]
        importance[f"{name.upper()}_selection_folds"] = [
            selection_count[name].get(marker, 0) for marker in importance["Marker"]
        ]
    coordinates = importance["Marker"].str.extract(
        r"chr(\d+)_(\d+)_([ACGT])_([ACGT])"
    )
    importance["Chr"] = coordinates[0].astype(int)
    importance["Pos"] = coordinates[1].astype(int)
    importance.sort_values("RF_mean_importance", ascending=False).to_csv(
        RUN_DIR / "genomewide_ml_marker_importance.tsv", sep="\t", index=False
    )
    metrics = {}
    for name in ("rf", "gb"):
        metrics[name] = {
            "oof_r2": float(r2_score(y, predictions[name])),
            "oof_rmse": float(math.sqrt(mean_squared_error(y, predictions[name]))),
            "oof_mae": float(mean_absolute_error(y, predictions[name])),
        }
    metrics["pc_only"] = {
        "oof_r2": float(r2_score(y, pc_predictions)),
        "oof_rmse": float(math.sqrt(mean_squared_error(y, pc_predictions))),
        "oof_mae": float(mean_absolute_error(y, pc_predictions)),
    }
    metrics["n_folds"] = N_OUTER_FOLDS
    metrics["candidate_screen"] = (
        "Training-fold RF and GB feature-importance screening within each chromosome; "
        "no GWAS summary statistics or post-hoc causal labels used"
    )
    metrics["maximum_candidates_per_fold"] = (
        2 * len(CHROMS) * TOP_PER_CHR_PER_MODEL
    )
    return {"metrics": metrics, "importance": importance, "predictions": predictions,
            "pc_predictions": pc_predictions}


def posthoc_causal_comparison(gwas_top: pd.DataFrame,
                              ml_importance: pd.DataFrame) -> pd.DataFrame:
    truth = pd.read_csv(PILOT_DIR / "causal_truth_POSTHOC_ONLY.csv")
    gwas_all = gwas_top.copy().reset_index(drop=True)
    gwas_all["GWAS_rank"] = np.arange(1, len(gwas_all) + 1)
    rf_top = ml_importance.nlargest(TOP_REPORT, "RF_mean_importance").reset_index(drop=True)
    rf_top["RF_rank"] = np.arange(1, len(rf_top) + 1)
    gb_top = ml_importance.nlargest(TOP_REPORT, "GB_mean_importance").reset_index(drop=True)
    gb_top["GB_rank"] = np.arange(1, len(gb_top) + 1)
    rows = []
    for _, locus in truth.iterrows():
        chrom, pos = int(locus["chrom"]), int(locus["pos"])
        locus_result = {
            "rsid": locus["rsid"], "gene": locus["gene"],
            "Chr": chrom, "causal_Pos": pos,
        }
        for label, table, statistic, rank_col in (
            ("GWAS", gwas_all, "p_PC_adjusted", "GWAS_rank"),
            ("RF", rf_top, "RF_mean_importance", "RF_rank"),
            ("GB", gb_top, "GB_mean_importance", "GB_rank"),
        ):
            same_chr = table.loc[table["Chr"] == chrom].copy()
            if same_chr.empty:
                locus_result[f"{label}_rank_nearest_within_100kb"] = "not in top list"
                locus_result[f"{label}_nearest_distance_bp"] = np.nan
                continue
            same_chr["distance_bp"] = (same_chr["Pos"] - pos).abs()
            nearest = same_chr.sort_values("distance_bp").iloc[0]
            locus_result[f"{label}_rank_nearest_within_100kb"] = (
                float(nearest[rank_col])
                if nearest["distance_bp"] <= 100_000 else "not in top list"
            )
            locus_result[f"{label}_nearest_distance_bp"] = int(nearest["distance_bp"])
            locus_result[f"{label}_nearest_marker"] = (
                f"chr{chrom}_{int(nearest['Pos'])}_{nearest.get('Ref','')}_{nearest.get('Alt','')}"
            )
            locus_result[f"{label}_nearest_statistic"] = float(nearest[statistic])
        rows.append(locus_result)
    result = pd.DataFrame(rows)
    result.to_csv(RUN_DIR / "causal_loci_posthoc.tsv", sep="\t", index=False)
    return result


def write_plots(gwas_path: Path, ml: pd.DataFrame) -> None:
    gwas_parts = []
    for frame in pd.read_csv(gwas_path, sep="\t", chunksize=150_000):
        gwas_parts.append(frame[["Chr", "Pos", "p_PC_adjusted", "BH_FDR"]])
    gwas = pd.concat(gwas_parts, ignore_index=True)
    gwas["logp"] = -np.log10(np.maximum(gwas["p_PC_adjusted"], 1e-300))
    offsets = {}
    running = 0
    tick_positions = []
    for chrom in CHROMS:
        offsets[chrom] = running
        subset = gwas.loc[gwas["Chr"] == chrom, "Pos"]
        if not subset.empty:
            tick_positions.append(running + subset.max() / 2)
            running += int(subset.max())
    gwas["x"] = gwas["Pos"] + gwas["Chr"].map(offsets)
    colors = ["#476a8a" if chrom % 2 else "#76a5af" for chrom in gwas["Chr"]]
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.scatter(gwas["x"], gwas["logp"], c=colors, s=1.0, linewidths=0, rasterized=True)
    bonf = -np.log10(0.05 / len(gwas))
    ax.axhline(bonf, color="#b5413e", linestyle="--", linewidth=1,
               label=f"Bonferroni 0.05 / {len(gwas):,}")
    ax.set_xticks(tick_positions, [str(c) for c in CHROMS])
    ax.set_xlabel("Chromosome (GRCh38)")
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_title("PC-adjusted genome-wide scan of the simulated pigmentation phenotype")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RUN_DIR / "genomewide_gwas_manhattan.png", dpi=220)
    plt.close(fig)

    pvalues = gwas["p_PC_adjusted"].to_numpy()
    observed = -np.log10(np.maximum(np.sort(pvalues), 1e-300))
    expected = -np.log10((np.arange(1, len(pvalues) + 1) - 0.5) / len(pvalues))
    chi2 = stats.chi2.isf(np.maximum(pvalues, 1e-300), df=1)
    lambda_gc = float(np.median(chi2) / stats.chi2.ppf(0.5, 1))
    zoom_limit = max(float(np.nanpercentile(observed, 99.9)), 3.0)
    zoom_mask = (expected <= zoom_limit) & (observed <= zoom_limit)
    full_x_max = float(expected.max())
    full_y_max = float(observed.max())
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12, 5.5))
    ax_full.scatter(
        expected, observed, s=1, alpha=0.35, color="#365f91", rasterized=True
    )
    ax_full.plot(
        [0, full_x_max], [0, full_x_max],
        color="#b5413e", linestyle="--", linewidth=1,
    )
    ax_full.set_xlim(0, full_x_max)
    ax_full.set_ylim(0, full_y_max * 1.02)
    ax_full.set_title("Full range")
    ax_zoom.scatter(
        expected[zoom_mask], observed[zoom_mask],
        s=1, alpha=0.35, color="#365f91", rasterized=True,
    )
    ax_zoom.plot(
        [0, zoom_limit], [0, zoom_limit],
        color="#b5413e", linestyle="--", linewidth=1,
    )
    ax_zoom.set_xlim(0, zoom_limit)
    ax_zoom.set_ylim(0, zoom_limit)
    ax_zoom.set_title("Bulk (upper 0.1% omitted)")
    fig.suptitle(f"Genome-wide QQ plot (λGC = {lambda_gc:.3f})")
    for ax in (ax_full, ax_zoom):
        ax.set_xlabel("Expected −log10(p)")
        ax.set_ylabel("Observed −log10(p)")
    fig.tight_layout()
    fig.savefig(RUN_DIR / "genomewide_gwas_qq.png", dpi=220)
    plt.close(fig)
    bonferroni_p = 0.05 / len(gwas)
    pd.DataFrame({
        "n_markers": [len(gwas)],
        "lambda_gc": [lambda_gc],
        "bonferroni_p": [bonferroni_p],
        "bonferroni_log10p": [bonf],
        "n_bonferroni_significant": [
            int((gwas["p_PC_adjusted"] < bonferroni_p).sum())
        ],
        "n_bh_fdr_0_05": [int((gwas["BH_FDR"] < 0.05).sum())],
    }).to_csv(
        RUN_DIR / "gwas_scan_summary.csv", index=False
    )

    ml = ml.copy()
    ml["x"] = ml["Pos"] + ml["Chr"].map(offsets)
    for column, title, filename, color in (
        ("RF_mean_importance", "Random Forest", "genomewide_rf_importance.png", "#b34a43"),
        ("GB_mean_importance", "Gradient Boosting", "genomewide_gb_importance.png", "#397b6a"),
    ):
        fig, ax = plt.subplots(figsize=(14, 4.5))
        ax.scatter(ml["x"], ml[column], c="#9aa4ad", s=7, alpha=0.6, linewidths=0)
        top = ml.nlargest(TOP_REPORT, column)
        ax.scatter(top["x"], top[column], c=color, s=15, linewidths=0, label="top 50")
        ax.set_xticks(tick_positions, [str(c) for c in CHROMS])
        ax.set_xlabel("Chromosome (GRCh38)")
        ax.set_ylabel("Mean fold-averaged feature importance")
        ax.set_title(f"{title} importance across genome-wide finalists (not p-values)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(RUN_DIR / filename, dpi=220)
        plt.close(fig)

    preds = pd.read_csv(RUN_DIR / "nested_cv_oof_predictions.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, title in zip(
        axes, ["rf_oof_prediction", "gb_oof_prediction", "pc_only_oof_prediction"],
        ["Random Forest", "Gradient Boosting", "PC-only baseline"],
    ):
        ax.scatter(preds["y_simulated"], preds[key], s=13, alpha=0.7, color="#365f91")
        ax.plot([-3, 3], [-3, 3], "--", color="#b5413e", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Simulated phenotype")
        ax.set_ylabel("Out-of-fold prediction")
    fig.tight_layout()
    fig.savefig(RUN_DIR / "nested_cv_predicted_vs_simulated.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "extract", "analyze"], default="all")
    args = parser.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    samples = load_samples()
    phenotype = pd.read_csv(PILOT_DIR / "phenotypes.csv")
    if phenotype["sample"].tolist() != samples:
        raise RuntimeError("Pilot phenotype sample order does not match the EUR panel.")
    y = phenotype["pigmentation_score_simulated"].to_numpy(dtype=float)
    phenotype.to_csv(RUN_DIR / "phenotype_used.csv", index=False)

    if args.stage in ("all", "extract"):
        manifest = extract_genome(samples)
        if args.stage == "extract":
            return
    else:
        manifest_path = RUN_DIR / "extraction_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("Run --stage extract before --stage analyze.")
        manifest = json.loads(manifest_path.read_text())

    pcs_frame = pd.read_csv(RUN_DIR / "ancestry_pcs.csv")
    if pcs_frame["sample"].tolist() != samples:
        raise RuntimeError("PCA sample order does not match the EUR sample panel.")
    pcs = pcs_frame.drop(columns="sample").to_numpy(dtype=float)
    start = time.time()
    gwas_top = run_gwas(manifest, y, pcs)
    ml_result = run_nested_ml(manifest, y, pcs)
    ml_result["importance"].sort_values(
        "RF_mean_importance", ascending=False
    ).to_csv(RUN_DIR / "genomewide_ml_marker_importance.tsv", sep="\t", index=False)
    posthoc_causal_comparison(gwas_top, ml_result["importance"])
    write_plots(
        RUN_DIR / "genomewide_gwas_summary_stats.tsv.gz",
        ml_result["importance"],
    )

    summary = {
        "n_samples": N_EUR,
        "n_autosomal_common_snvs": manifest["n_markers"],
        "maf_min": MAF_MIN,
        "n_pcs": pcs.shape[1],
        "pc_only_oof_r2": ml_result["metrics"]["pc_only"]["oof_r2"],
        "rf_oof": ml_result["metrics"]["rf"],
        "gb_oof": ml_result["metrics"]["gb"],
        "n_outer_folds": N_OUTER_FOLDS,
        "ml_screen": ml_result["metrics"]["candidate_screen"],
        "maximum_candidates_per_fold": ml_result["metrics"]["maximum_candidates_per_fold"],
        "runtime_analysis_seconds": time.time() - start,
        "phenotype_source": str(PILOT_DIR / "phenotypes.csv"),
        "phenotype_is_measured": False,
        "causal_truth_used_in_training": False,
    }
    (RUN_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
