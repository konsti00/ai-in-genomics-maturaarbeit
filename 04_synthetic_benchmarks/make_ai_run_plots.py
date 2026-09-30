#!/usr/bin/env python3
"""
Generates diagnostic plots for the ML runs on simulated outcomes
(04_synthetic_benchmarks/blind_simulation,
04_synthetic_benchmarks/context_calibrated_simulation and 05_human_pilot).
These runs use ML feature importance
(Random Forest / Gradient Boosting), not GLM/MLM test-statistic p-values,
so there are no p-values to build a classical Manhattan/QQ plot from.
This script builds the closest honest analogs:

  1. Manhattan-style plot of RF/GB mean importance across the genome,
     with the planted markers highlighted.
  2. Causal-recovery "QQ-style" plot: observed vs. expected-under-random
     rank quantiles of the causal markers among all screened candidates.
     (This substitutes for a p-value QQ plot in this no-p-value pipeline.)
  3. Out-of-fold predicted-vs-actual scatter plot (RF and GB).
  4. Per-fold R^2 / RMSE comparison bar chart (RF vs GB).

Each plot is saved as a PNG directly inside the corresponding run folder,
prefixed with the run name so files from different runs are never
confused with each other.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
PILOT_DIR = os.path.join(BASE, "..", "05_human_pilot")

RUNS = [
    {
        "dir": os.path.join(BASE, "blind_simulation"),
        "tag": "blind_simulation",
        "title": "Blind simulation (FT10-calibrated genome, no phenotype knowledge used)",
        "short": "Blind simulation",
        "marker_col": "Marker", "chr_col": "Chr", "pos_col": "Pos",
        "causal_marker_col": "Marker", "causal_chr_col": "Chr", "causal_pos_col": "Pos",
        "causal_label_col": None,
        "fold_r2_cols": ("rf_r2", "gb_r2"), "fold_rmse_cols": ("rf_rmse", "gb_rmse"),
    },
    {
        "dir": os.path.join(BASE, "context_calibrated_simulation"),
        "tag": "context_calibrated_simulation",
        "title": "Context-calibrated simulation (summary statistics used only to generate phenotype)",
        "short": "Context-calibrated simulation",
        "marker_col": "Marker", "chr_col": "Chr", "pos_col": "Pos",
        "causal_marker_col": "Marker", "causal_chr_col": "Chr", "causal_pos_col": "Pos",
        "causal_label_col": None,
        "fold_r2_cols": ("rf_r2", "gb_r2"), "fold_rmse_cols": ("rf_rmse", "gb_rmse"),
    },
    {
        "dir": PILOT_DIR,
        "tag": "human_eyecolor_pilot",
        "title": "Human pigmentation pilot (1000 Genomes EUR samples, simulated outcome)",
        "short": "Human pigmentation pilot",
        "marker_col": "Marker", "chr_col": "Chr", "pos_col": "Pos",
        "causal_marker_col": "Marker", "causal_chr_col": "chrom", "causal_pos_col": "pos",
        "causal_label_col": "gene", "causal_rsid_source_col": "rsid",
        "fold_r2_cols": ("rf_r2_fold", "gb_r2_fold"), "fold_rmse_cols": (None, None),
    },
]

CHR_COLORS = ["#3b6fa0", "#8faed1"]  # alternating shades for adjacent chromosomes


def chrom_offsets(chr_lengths):
    """chr_lengths: dict {chr:int -> length_bp-ish (max pos works too)}. Returns cumulative offsets."""
    offsets = {}
    cum = 0
    for c in sorted(chr_lengths, key=lambda x: int(x)):
        offsets[c] = cum
        cum += chr_lengths[c]
    return offsets, cum


def load_chrom_lengths(run_dir, df_all_positions):
    """Prefer calibration_summary.json / simulation_manifest.json; fall back to max observed pos."""
    for fname in ("calibration_summary.json", "simulation_manifest.json"):
        path = os.path.join(run_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                manifest = json.load(f)
            if "chromosomes" in manifest:
                return {int(k): v["length_bp"] for k, v in manifest["chromosomes"].items()}
    # Fallback: use max position per chromosome observed in the data, padded 5%
    lengths = df_all_positions.groupby("Chr")["Pos"].max().to_dict()
    return {int(k): int(v * 1.05) for k, v in lengths.items()}


def manhattan_plot(run, screened, causal_df):
    chr_lengths = load_chrom_lengths(run["dir"], screened.rename(columns={run["chr_col"]: "Chr", run["pos_col"]: "Pos"}))
    offsets, genome_len = chrom_offsets(chr_lengths)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for ax, (model_col, model_name) in zip(
        axes, [("RF_mean_importance", "Random Forest"), ("GB_mean_importance", "Gradient Boosting")]
    ):
        df = screened.copy()
        df["cum_pos"] = df.apply(lambda r: offsets[int(r[run["chr_col"]])] + r[run["pos_col"]], axis=1)
        df["color"] = df[run["chr_col"]].apply(lambda c: CHR_COLORS[int(c) % 2])
        ax.scatter(df["cum_pos"], df[model_col], c=df["color"], s=6, alpha=0.6, linewidths=0, rasterized=True)

        # top-50 threshold line
        thresh = df[model_col].nlargest(50).min()
        ax.axhline(thresh, color="gray", linestyle="--", linewidth=1, label="Top-50 importance threshold")

        # highlight causal markers
        cdf = causal_df.copy()
        if run["causal_marker_col"] not in cdf.columns:
            cdf[run["causal_marker_col"]] = (
                "Chr" + cdf[run["causal_chr_col"]].astype(int).astype(str) + "_" + cdf[run["causal_pos_col"]].astype(int).astype(str)
            )
        cdf["cum_pos"] = cdf.apply(
            lambda r: offsets.get(int(r[run["causal_chr_col"]]), 0) + r[run["causal_pos_col"]], axis=1
        )
        merged = cdf.merge(
            df[[run["marker_col"], model_col]], left_on=run["causal_marker_col"], right_on=run["marker_col"], how="left"
        )
        ax.scatter(
            merged["cum_pos"], merged[model_col].fillna(0), marker="*", s=170,
            facecolor="none", edgecolor="red", linewidths=1.5, label="Truly causal marker", zorder=5,
        )
        if run["causal_label_col"] and run["causal_label_col"] in cdf.columns:
            for _, row in merged.iterrows():
                yval = row[model_col] if pd.notna(row[model_col]) else 0
                ax.annotate(
                    row[run["causal_label_col"]], (row["cum_pos"], yval), fontsize=7,
                    textcoords="offset points", xytext=(4, 4), color="darkred",
                )

        ax.set_ylabel(f"{model_name}\nmean importance")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # chromosome tick labels at the midpoint of each chromosome
    mids = []
    labels = []
    sorted_chr = sorted(chr_lengths, key=lambda x: int(x))
    for c in sorted_chr:
        mids.append(offsets[c] + chr_lengths[c] / 2)
        labels.append(str(c))
    axes[-1].set_xticks(mids)
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel("Chromosome")
    fig.suptitle(f"Importance across the genome: {run['title']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(run["dir"], f"{run['tag']}_manhattan_importance.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def causal_recovery_qq_plot(run, causal_recovery, n_candidates):
    fig, ax = plt.subplots(figsize=(7, 7))
    styles = {"Random Forest": ("#1b7837", "o"), "Gradient Boosting": ("#762a83", "^")}
    any_data = False
    for model_name, (color, marker) in styles.items():
        sub = causal_recovery[causal_recovery["model"] == model_name].copy()
        if "final_importance_rank" not in sub.columns:
            continue
        ranks = sub["final_importance_rank"].dropna()
        if len(ranks) == 0:
            continue
        any_data = True
        k = len(ranks)
        observed_q = np.sort(ranks.values) / n_candidates
        expected_q = (np.arange(1, k + 1)) / (k + 1)
        ax.scatter(
            expected_q, observed_q, color=color, marker=marker, s=70, alpha=0.75,
            edgecolors="black", linewidths=0.5, label=f"{model_name} (n={k} planted markers)", zorder=3,
        )

    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="Expected under a random ranking")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Expected rank quantile if importance were random (uniform)")
    ax.set_ylabel("Observed rank quantile of planted markers\n(among all screened candidates)")
    ax.set_title(
        "\n".join([
            f"Ranks of planted markers: {run['short']}",
            "(points below the diagonal = better than a random ranking)",
        ]),
        fontsize=10,
    )
    if any_data:
        ax.legend(loc="upper left", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No causal markers were ever\nselected into a finalist model\n(all ranks missing)",
                 ha="center", va="center", fontsize=10, color="gray")
    fig.tight_layout()
    out = os.path.join(run["dir"], f"{run['tag']}_causal_recovery_qq.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def predicted_vs_actual_plot(run, oof):
    cols = oof.columns
    y_true_col = [c for c in cols if c.lower() in ("ft10_simulated", "y_true")][0]
    rf_col = [c for c in cols if "rf" in c.lower()][0]
    gb_col = [c for c in cols if "gb" in c.lower()][0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, col, name, color in zip(axes, [rf_col, gb_col], ["Random Forest", "Gradient Boosting"], ["#1b7837", "#762a83"]):
        x = oof[y_true_col]
        y = oof[col]
        ax.scatter(x, y, s=10, alpha=0.5, color=color, linewidths=0)
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, color="gray", linestyle="--", linewidth=1, label="y = x (perfect prediction)")
        r2 = 1 - np.sum((x - y) ** 2) / np.sum((x - x.mean()) ** 2)
        ax.set_title(f"{name}\nout-of-fold R\u00b2 = {r2:.3f}")
        ax.set_xlabel("True phenotype (held-out folds)")
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("Predicted phenotype (out-of-fold)")
    fig.suptitle(f"Out-of-fold predicted vs. actual: {run['title']}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(run["dir"], f"{run['tag']}_predicted_vs_actual.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fold_metrics_plot(run, fold_metrics):
    r2_rf_col, r2_gb_col = run["fold_r2_cols"]
    if r2_rf_col not in fold_metrics.columns or r2_gb_col not in fold_metrics.columns:
        return None
    folds = fold_metrics["fold"].astype(str)
    x = np.arange(len(folds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(x - width / 2, fold_metrics[r2_rf_col], width, label="Random Forest", color="#1b7837")
    ax.bar(x + width / 2, fold_metrics[r2_gb_col], width, label="Gradient Boosting", color="#762a83")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.set_xlabel("Outer cross-validation fold")
    ax.set_ylabel("R\u00b2 (held-out fold)")
    ax.set_title(f"Per-fold model R\u00b2: {run['title']}", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(run["dir"], f"{run['tag']}_fold_r2.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main():
    for run in RUNS:
        print(f"=== {run['tag']} ===")
        screened = pd.read_csv(os.path.join(run["dir"], "screened_candidates_and_importance.csv"))
        causal_df = pd.read_csv(os.path.join(run["dir"], "causal_truth_POSTHOC_ONLY.csv"))
        causal_recovery = pd.read_csv(os.path.join(run["dir"], "causal_recovery_posthoc.csv"))
        oof = pd.read_csv(os.path.join(run["dir"], "out_of_fold_predictions.csv"))
        fold_metrics = pd.read_csv(os.path.join(run["dir"], "fold_metrics.csv"))
        n_candidates = screened.shape[0]

        p1 = manhattan_plot(run, screened, causal_df)
        print("  wrote", p1)
        p2 = causal_recovery_qq_plot(run, causal_recovery, n_candidates)
        print("  wrote", p2)
        p3 = predicted_vs_actual_plot(run, oof)
        print("  wrote", p3)
        p4 = fold_metrics_plot(run, fold_metrics)
        if p4:
            print("  wrote", p4)
        else:
            print("  skipped fold R2 plot (columns not found)")


if __name__ == "__main__":
    main()
