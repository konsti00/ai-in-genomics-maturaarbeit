import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

files = ["ft10_chr1_block022", "ft10_chr5_block021", "ft10_chr5_block017", "ft10_chr3_block019"]
frames = []
for f in files:
    df = pd.read_csv(f"ft16_gapit_results/{f}_GWAS.csv")
    df.columns = [c.strip() for c in df.columns]
    frames.append(df)
ft16 = pd.concat(frames, ignore_index=True)
ft16["mlog10p"] = -np.log10(ft16["P.value"].clip(lower=1e-300))

# Chr5 18.59Mb region zoom (strongest FT16 signal in the shared set)
sub = ft16[(ft16["Chromosome"] == 5) & (ft16["Position"].between(18_570_000, 18_610_000))]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.scatter(sub["Position"], sub["mlog10p"], s=6, color="#2f6b3d", alpha=0.6)
bonf_line = -np.log10(0.05 / len(ft16))
ax.axhline(bonf_line, color="#8b2f2f", linestyle="--", linewidth=1, label="Bonferroni 0.05 (this block set)")
ax.set_xlabel("Chromosome 5 position")
ax.set_ylabel("-log10(P)")
ax.set_title("FT16 regional association near Chr5 ~18.59 Mb (replication check)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("ft16_zoom_chr5_18.59Mb.png", dpi=200)
plt.close(fig)
print("wrote ft16_zoom_chr5_18.59Mb.png, n=", len(sub))
