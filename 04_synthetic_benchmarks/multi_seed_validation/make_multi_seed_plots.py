#!/usr/bin/env python3
"""
Generates aggregate diagnostic plots for the multi_seed_validation folder
(5 blind-scenario seeds + 5 context-calibrated-scenario seeds). Unlike the
single-run plots in make_ai_run_plots.py, per-seed Manhattan/QQ plots
would just repeat the same picture 10 times, so this script instead
focuses on what is actually informative here: how stable the model's
performance and causal-locus recovery are *across* random seeds within
each scenario. Reads the raw per-seed files directly (not just the
aggregate_results.csv summary stats) so the box/strip plots show real
seed-to-seed spread rather than only mean +/- sd.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
SCENARIOS = ["blind", "context_calibrated"]
SEEDS = [1, 2, 3, 4, 5]
MODEL_COLORS = {"Random Forest": "#1b7837", "Gradient Boosting": "#762a83"}


def collect_records():
    records = []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            folder = os.path.join(BASE, f"{scenario}_seed{seed}")
            with open(os.path.join(folder, "analysis_summary.json")) as f:
                summary = json.load(f)
            causal_recovery = pd.read_csv(os.path.join(folder, "causal_recovery_posthoc.csv"))
            n_causal = causal_recovery["causal_marker"].nunique()
            for model_key, model_name in [("rf", "Random Forest"), ("gb", "Gradient Boosting")]:
                n_top50 = causal_recovery[
                    (causal_recovery["model"] == model_name) & (causal_recovery["exactly_in_top50"] == True)
                ].shape[0]
                records.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "model": model_name,
                        "oof_r2": summary[f"{model_key}_oof_r2"],
                        "oof_rmse": summary[f"{model_key}_oof_rmse"],
                        "causal_in_top50": n_top50,
                        "n_causal": n_causal,
                    }
                )
    return pd.DataFrame(records)


def strip_box_plot(df, value_col, ylabel, title, out_path, hline=None, hline_label=None):
    scenarios = SCENARIOS
    models = list(MODEL_COLORS.keys())
    fig, ax = plt.subplots(figsize=(8, 5.5))

    positions = []
    labels = []
    pos = 0
    box_data = []
    box_colors = []
    for scenario in scenarios:
        for model in models:
            sub = df[(df["scenario"] == scenario) & (df["model"] == model)][value_col]
            box_data.append(sub.values)
            box_colors.append(MODEL_COLORS[model])
            positions.append(pos)
            labels.append(f"{scenario}\n{model}")
            pos += 1
        pos += 0.6  # gap between scenarios

    bp = ax.boxplot(
        box_data, positions=positions, widths=0.6, showfliers=False, patch_artist=True,
        medianprops=dict(color="black"),
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    # overlay individual seed points (jittered)
    rng = np.random.default_rng(0)
    for xpos, data, color in zip(positions, box_data, box_colors):
        jitter = rng.uniform(-0.12, 0.12, size=len(data))
        ax.scatter(np.full(len(data), xpos) + jitter, data, color=color, edgecolor="black",
                   linewidths=0.5, s=45, zorder=4)

    if hline is not None:
        ax.axhline(hline, color="gray", linestyle="--", linewidth=1, label=hline_label)
        ax.legend(fontsize=8, loc="best")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    df = collect_records()
    df.to_csv(os.path.join(BASE, "multi_seed_raw_per_seed_metrics.csv"), index=False)
    print("Collected per-seed records:\n", df)

    strip_box_plot(
        df, "oof_r2", "Out-of-fold R\u00b2",
        "Multi-seed stability — out-of-fold R\u00b2 across 5 seeds per scenario\n"
        "(box = seed distribution, dots = individual seeds)",
        os.path.join(BASE, "multi_seed_oof_r2_boxplot.png"),
    )
    print("wrote multi_seed_oof_r2_boxplot.png")

    n_causal = df["n_causal"].iloc[0]
    strip_box_plot(
        df, "causal_in_top50", f"Causal markers recovered in top-50\n(out of {n_causal} planted/known causal markers)",
        "Multi-seed stability — causal-locus recovery across 5 seeds per scenario\n"
        "(box = seed distribution, dots = individual seeds)",
        os.path.join(BASE, "multi_seed_causal_recovery_boxplot.png"),
        hline=n_causal * (50 / df["oof_r2"].count()) if False else None,
    )
    print("wrote multi_seed_causal_recovery_boxplot.png")

    # Combined summary panel: R2 and causal recovery side by side
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, value_col, ylabel in zip(
        axes, ["oof_r2", "causal_in_top50"],
        ["Out-of-fold R\u00b2", f"Causal markers recovered in top-50 (of {n_causal})"]
    ):
        positions = []
        labels = []
        pos = 0
        for scenario in SCENARIOS:
            for model in MODEL_COLORS:
                sub = df[(df["scenario"] == scenario) & (df["model"] == model)][value_col]
                mean = sub.mean()
                sd = sub.std()
                ax.bar(pos, mean, yerr=sd, width=0.6, color=MODEL_COLORS[model], alpha=0.7, capsize=4)
                rng = np.random.default_rng(1)
                jitter = rng.uniform(-0.12, 0.12, size=len(sub))
                ax.scatter(np.full(len(sub), pos) + jitter, sub, color="black", s=18, zorder=4)
                positions.append(pos)
                labels.append(f"{scenario}\n{model}")
                pos += 1
            pos += 0.6
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel)
    fig.suptitle("Multi-seed validation summary (mean \u00b1 SD across 5 seeds, dots = individual seeds)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(BASE, "multi_seed_summary_panel.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote multi_seed_summary_panel.png")


if __name__ == "__main__":
    main()
