"""
Build a genome-wide, systematically-thinned genotype matrix for AI-based
(machine-learning) GWAS analysis. This is a DATA PREPARATION step only:

- Reuses the already MAF-filtered (>=1%) SNP blocks extracted earlier from the
  1001 Genomes HDF5 matrix (pure data engineering, no association testing).
- Applies a fixed, association-blind systematic thinning (every Nth marker in
  genome order) purely to keep the machine-learning models computationally
  feasible. This is NOT a GWAS p-value/model-based marker selection.
- No kinship correction, no mixed model, no per-SNP hypothesis testing is
  used anywhere in this script or in the downstream ML training.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SOURCE_BASE = BASE.parents[1] / "02_arabidopsis_gwas"
BLOCK_DIR = SOURCE_BASE / "ft10_blocks"
PHENOTYPE_FILE = SOURCE_BASE / "ft10_gapit_phenotype.csv"
THIN_FACTOR = 10  # keep every 10th marker -> ~356k of ~3.56M SNPs

OUT_GENOTYPE = BASE / "ml_genotype_matrix.npz"
OUT_MARKERS = BASE / "ml_marker_map.csv"
OUT_PHENOTYPE = BASE / "ml_phenotype.csv"


def block_sort_key(path):
    match = re.search(r"chr(\d+)_block(\d+)", path.name)
    return tuple(map(int, match.groups())) if match else (999, 999)


def main():
    phenotype = pd.read_csv(PHENOTYPE_FILE)
    phenotype["Taxa"] = phenotype["Taxa"].astype(str)
    taxa_order = phenotype["Taxa"].tolist()

    block_files = sorted(BLOCK_DIR.glob("*.csv.gz"), key=block_sort_key)
    if not block_files:
        raise FileNotFoundError(f"No genotype blocks found in {BLOCK_DIR}")

    genotype_chunks = []
    marker_records = []

    for block_file in block_files:
        block_name = block_file.name.replace(".csv.gz", "")
        marker_file = BLOCK_DIR / f"{block_name}_markers.csv"

        raw = pd.read_csv(block_file)
        raw = raw.set_index("Marker")
        raw = raw.reindex(columns=taxa_order)

        markers = pd.read_csv(marker_file)
        markers = markers.set_index("Marker")

        # Systematic (association-blind) thinning: every Nth marker in
        # genome order within this block.
        keep_idx = np.arange(0, len(raw), THIN_FACTOR)
        thinned = raw.iloc[keep_idx]
        thinned_markers = markers.loc[thinned.index]

        genotype_chunks.append(thinned.T.astype(np.int8))
        marker_records.append(thinned_markers.reset_index())

        print(f"{block_name}: kept {len(thinned)} / {len(raw)} markers")

    genotype_matrix = pd.concat(genotype_chunks, axis=1)
    marker_map = pd.concat(marker_records, ignore_index=True)

    assert list(genotype_matrix.index) == taxa_order

    np.savez_compressed(
        OUT_GENOTYPE,
        X=genotype_matrix.to_numpy(dtype=np.int8),
        taxa=np.array(genotype_matrix.index, dtype=str),
        markers=np.array(genotype_matrix.columns, dtype=str),
    )
    marker_map.to_csv(OUT_MARKERS, index=False)
    phenotype.to_csv(OUT_PHENOTYPE, index=False)

    print(f"Final matrix shape (samples x markers): {genotype_matrix.shape}")
    print(f"Saved {OUT_GENOTYPE}, {OUT_MARKERS}, {OUT_PHENOTYPE}")


if __name__ == "__main__":
    main()
