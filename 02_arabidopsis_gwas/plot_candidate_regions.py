import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("ft10_FT10_GWAS_all.csv")

regions = [
    dict(chrom=1, lo=24_300_000, hi=24_370_000, gene="FT (AT1G65480)",
         gstart=24_331_373, gend=24_333_999, out="zoom_chr1_FT.png"),
    dict(chrom=5, lo=23_050_000, hi=23_330_000, gene="ZTL (AT5G57360)",
         gstart=23_231_129, gend=23_233_376, out="zoom_chr5_ZTL.png"),
]

for r in regions:
    sub = df[(df["Chromosome"] == r["chrom"]) & (df["Position"].between(r["lo"], r["hi"]))].copy()
    sub["mlog10p"] = -np.log10(sub["P.value"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(sub["Position"], sub["mlog10p"], s=6, color="#174a5b", alpha=0.6)
    ax.axvspan(r["gstart"], r["gend"], color="#d66b3d", alpha=0.3, label=r["gene"])
    ax.axhline(-np.log10(0.05 / len(df)), color="#8b2f2f", linestyle="--", linewidth=1, label="Bonferroni 0.05")
    ax.set_xlabel(f"Chromosome {r['chrom']} position")
    ax.set_ylabel("-log10(P)")
    ax.set_title(f"FT10 regional association near {r['gene']}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(r["out"], dpi=200)
    plt.close(fig)
    print("wrote", r["out"], "n_snps_plotted:", len(sub))
