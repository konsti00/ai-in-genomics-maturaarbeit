from pathlib import Path
import math
import re

import numpy as np
import pandas as pd

RESULT_DIR = Path("ft10_gapit_results")
MERGED_PATH = Path("ft10_FT10_GWAS_all.csv")
TOP_PATH = Path("ft10_FT10_GWAS_top_associations.csv")
SUMMARY_PATH = Path("ft10_FT10_GWAS_summary.txt")
MANHATTAN_PATH = Path("ft10_FT10_Manhattan.png")
QQ_PATH = Path("ft10_FT10_QQ.png")


def benjamini_hochberg(p_values):
    """Return BH-adjusted p-values in the original order."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values, kind="mergesort")
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def block_sort_key(path):
    match = re.search(r"chr(\\d+)_block(\\d+)", path.name)
    return tuple(map(int, match.groups())) if match else (999, 999)


def main():
    files = sorted(RESULT_DIR.glob("*_GWAS.csv"), key=block_sort_key)
    if not files:
        raise FileNotFoundError(f"No GWAS result files found in {RESULT_DIR}")

    tables = []
    for path in files:
        table = pd.read_csv(path)
        table.columns = [str(column).strip() for column in table.columns]
        required = {"SNP", "Chromosome", "Position", "P.value"}
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        table["source_block"] = path.stem.replace("_GWAS", "")
        tables.append(table)

    results = pd.concat(tables, ignore_index=True)
    results["Chromosome"] = pd.to_numeric(results["Chromosome"], errors="raise").astype(int)
    results["Position"] = pd.to_numeric(results["Position"], errors="raise").astype(int)
    results["P.value"] = pd.to_numeric(results["P.value"], errors="coerce")
    results = results[np.isfinite(results["P.value"])].copy()
    results["P.value"] = results["P.value"].clip(lower=np.finfo(float).tiny, upper=1.0)

    results["BH_FDR"] = benjamini_hochberg(results["P.value"].to_numpy())
    results["Bonferroni"] = np.minimum(results["P.value"] * len(results), 1.0)
    results["minus_log10_P"] = -np.log10(results["P.value"])
    results = results.sort_values(["Chromosome", "Position", "SNP"], kind="mergesort").reset_index(drop=True)
    results.to_csv(MERGED_PATH, index=False)

    top = results.sort_values(["P.value", "BH_FDR", "Chromosome", "Position"], kind="mergesort").head(100)
    top.to_csv(TOP_PATH, index=False)

    try:
        import matplotlib.pyplot as plt

        plot_data = results.sort_values(["Chromosome", "Position"], kind="mergesort").copy()
        plot_data["plot_x"] = np.nan
        offset = 0
        ticks = []
        labels = []
        for chromosome, group in plot_data.groupby("Chromosome", sort=True):
            indices = group.index
            plot_data.loc[indices, "plot_x"] = group["Position"].to_numpy() + offset
            ticks.append(offset + group["Position"].max() / 2)
            labels.append(str(chromosome))
            offset += group["Position"].max()

        fig, axis = plt.subplots(figsize=(14, 5.5))
        for index, (chromosome, group) in enumerate(plot_data.groupby("Chromosome", sort=True)):
            axis.scatter(group["plot_x"], group["minus_log10_P"], s=2, alpha=0.55, color=("#174a5b", "#d66b3d")[index % 2], linewidths=0)
        genome_wide = -math.log10(0.05 / len(results))
        axis.axhline(genome_wide, color="#8b2f2f", linestyle="--", linewidth=1, label="Bonferroni 0.05")
        axis.set_xticks(ticks, labels)
        axis.set_xlabel("Chromosome")
        axis.set_ylabel("-log10(P)")
        axis.set_title("FT10 GAPIT MLM genome-wide association")
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(MANHATTAN_PATH, dpi=220)
        plt.close(fig)

        observed = np.sort(results["P.value"].to_numpy())
        expected = -np.log10((np.arange(1, len(observed) + 1) - 0.5) / len(observed))
        observed_log = -np.log10(observed)
        limit = max(expected.max(), observed_log.max())
        fig, axis = plt.subplots(figsize=(5.8, 5.8))
        axis.scatter(expected, observed_log, s=4, alpha=0.45, color="#174a5b", linewidths=0)
        axis.plot([0, limit], [0, limit], color="#8b2f2f", linestyle="--", linewidth=1)
        axis.set_xlabel("Expected -log10(P)")
        axis.set_ylabel("Observed -log10(P)")
        axis.set_title("FT10 QQ plot")
        fig.tight_layout()
        fig.savefig(QQ_PATH, dpi=220)
        plt.close(fig)
    except ImportError:
        SUMMARY_PATH.write_text("Plot generation skipped because matplotlib is unavailable.\n")

    significant_fdr = int((results["BH_FDR"] < 0.05).sum())
    significant_bonferroni = int((results["Bonferroni"] < 0.05).sum())
    minimum = results.iloc[results["P.value"].argmin()]
    summary = [
        f"Block files merged: {len(files)}",
        f"Markers tested: {len(results):,}",
        f"Unique SNP IDs: {results['SNP'].nunique():,}",
        f"BH FDR < 0.05: {significant_fdr:,}",
        f"Bonferroni < 0.05: {significant_bonferroni:,}",
        f"Smallest P.value: {minimum['P.value']:.6g}",
        f"Top SNP: {minimum['SNP']} (Chr {int(minimum['Chromosome'])}, position {int(minimum['Position'])})",
        f"Top SNP BH_FDR: {minimum['BH_FDR']:.6g}",
        f"Top SNP Bonferroni: {minimum['Bonferroni']:.6g}",
    ]
    SUMMARY_PATH.write_text("\n".join(summary) + "\n")
    print("\n".join(summary))
    print(f"Wrote {MERGED_PATH}, {TOP_PATH}, {SUMMARY_PATH}, {MANHATTAN_PATH}, and {QQ_PATH}")


if __name__ == "__main__":
    main()
