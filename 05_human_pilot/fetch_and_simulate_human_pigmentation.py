#!/usr/bin/env python3
"""Fetch real public human genotypes (1000 Genomes Project, GRCh38 high-coverage
call set) for four selected pigmentation-associated SNPs plus a genome-wide
background screening pool, restricted to the 503 samples of the European
(EUR) superpopulation, and simulate a quantitative pigmentation score for
those samples.

Data source and license note
-----------------------------
All genotype data used here comes from the 1000 Genomes Project 30x
high-coverage call set, which is fully public, open-access, and explicitly
released for unrestricted research reuse (no data-access agreement and no
individual-level phenotype data). No measured phenotype exists for these
samples: the outcome analysed downstream is entirely simulated.

Simulation design
-----------------
Each of the four selected variants enters the score additively as a
standardized allele dosage with a randomly assigned effect direction. The
weights are simulation settings: 0.70 for rs12913832 (HERC2/OCA2), which is
strongly associated with blue-brown eye colour (Sturm et al., 2008), and
illustrative weights of 0.03 for each of the three other variants. The
remaining 0.21 is random noise.

Circularity note (read before interpreting results)
----------------------------------------------------
Because the phenotype is generated from these four variants, any method
"recovering" them is an expected check against a known ground truth, not a
biological discovery or a replication of published associations.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

OUT_DIR = Path(__file__).resolve().parent
GENOTYPE_DIR = OUT_DIR / "genotype_blocks"
VCF_BASE = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
    "1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/"
    "1kGP_high_coverage_Illumina.chr{chrom}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
)
PANEL_PATH = OUT_DIR / "1000genomes_sample_panel.tsv"
MAF_MIN = 0.05
RANDOM_SEED = 20260401

# GRCh38 primary-assembly autosome lengths (bp), standard reference values.
CHROM_LENGTHS = {
    1: 248_956_422, 2: 242_193_529, 3: 198_295_559, 4: 190_214_555,
    5: 181_538_259, 6: 170_805_979, 7: 159_345_973, 8: 145_138_636,
    9: 138_394_717, 10: 133_797_422, 11: 135_086_622, 12: 133_275_309,
    13: 114_364_328, 14: 107_043_718, 15: 101_991_189, 16: 90_338_345,
    17: 83_257_441, 18: 80_373_285, 19: 58_617_616, 20: 64_444_167,
    21: 46_709_983, 22: 50_818_468,
}

# Four selected pigmentation-associated variants (GRCh38 coordinates verified
# against Ensembl's variation REST API), each defining a 300kb window.
# "role" is a descriptive label only; all variants enter the score additively.
CAUSAL_LOCI = [
    {"name": "rs12913832", "gene": "HERC2/OCA2 enhancer", "chrom": 15, "pos": 28_120_472,
     "role": "major", "target_var_fraction": 0.70,
     "note": "Largest simulated weight; strongly associated with blue-brown eye colour (Sturm et al., 2008)."},
    {"name": "rs16891982", "gene": "SLC45A2", "chrom": 5, "pos": 33_951_588,
     "role": "minor", "target_var_fraction": 0.03,
     "note": "Illustrative small simulated weight."},
    {"name": "rs1042602", "gene": "TYR", "chrom": 11, "pos": 89_178_528,
     "role": "minor", "target_var_fraction": 0.03,
     "note": "Illustrative small simulated weight."},
    {"name": "rs12203592", "gene": "IRF4", "chrom": 6, "pos": 396_321,
     "role": "minor", "target_var_fraction": 0.03,
     "note": "Illustrative small simulated weight."},
]
CAUSAL_WINDOW_HALF_WIDTH = 150_000
N_BACKGROUND_WINDOWS_PER_CHROM = 1
BACKGROUND_WINDOW_WIDTH = 500_000
BACKGROUND_OFFSET_FRACTION = 0.30  # keeps background windows off acrocentric
# chromosomes' short arms and away from most centromeres.
TARGET_RESIDUAL_VAR_FRACTION = 1.0 - sum(l["target_var_fraction"] for l in CAUSAL_LOCI)


def load_eur_samples() -> list[str]:
    panel = pd.read_csv(PANEL_PATH, sep="\t")
    eur = panel.loc[panel["super_pop"] == "EUR", "sample"].tolist()
    if len(eur) != 503:
        raise RuntimeError(f"Expected 503 EUR samples, found {len(eur)}.")
    return eur


def build_window_list() -> list[dict]:
    windows = []
    for locus in CAUSAL_LOCI:
        start = max(1, locus["pos"] - CAUSAL_WINDOW_HALF_WIDTH)
        end = locus["pos"] + CAUSAL_WINDOW_HALF_WIDTH
        windows.append(
            {
                "block_id": f"causal_{locus['name']}",
                "chrom": locus["chrom"],
                "start": start,
                "end": end,
                "kind": "causal_locus",
                "causal_snp": locus["name"],
            }
        )
    for chrom, length in CHROM_LENGTHS.items():
        start = int(length * BACKGROUND_OFFSET_FRACTION)
        end = start + BACKGROUND_WINDOW_WIDTH
        windows.append(
            {
                "block_id": f"background_chr{chrom}_{start}",
                "chrom": chrom,
                "start": start,
                "end": end,
                "kind": "background",
                "causal_snp": None,
            }
        )
    return windows


def parse_genotype_dosage(gt_field: str) -> int | None:
    """Convert a phased/unphased diploid GT string (e.g. '0|1', '1/1') to an
    allele dosage of the ALT allele (0, 1, or 2). Returns None if missing or
    multi-allelic (already excluded upstream by REF/ALT length checks, but
    guarded here too)."""
    alleles = gt_field.replace("|", "/").split("/")
    if len(alleles) != 2 or "." in alleles:
        return None
    try:
        a, b = int(alleles[0]), int(alleles[1])
    except ValueError:
        return None
    if a not in (0, 1) or b not in (0, 1):
        return None
    return a + b


def fetch_window(chrom: int, start: int, end: int, sample_columns: list[str],
                  sample_index: list[int]) -> tuple[np.ndarray, pd.DataFrame]:
    url = VCF_BASE.format(chrom=chrom)
    tbx = pysam.TabixFile(url)
    rows = []
    marker_rows = []
    for rec in tbx.fetch(f"chr{chrom}", start, end):
        fields = rec.split("\t")
        ref, alt = fields[3], fields[4]
        if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT":
            continue  # keep biallelic SNVs only, matching the plant-analysis filter
        pos = int(fields[1])
        genotype_fields = fields[9:]
        dosages = np.full(len(sample_index), -1, dtype=np.int8)
        ok = True
        for out_i, sample_i in enumerate(sample_index):
            dosage = parse_genotype_dosage(genotype_fields[sample_i])
            if dosage is None:
                ok = False
                break
            dosages[out_i] = dosage
        if not ok:
            continue
        maf = dosages.mean() / 2.0
        maf = min(maf, 1.0 - maf)
        if maf < MAF_MIN:
            continue
        rows.append(dosages)
        marker_rows.append(
            {
                "Chr": chrom,
                "Pos": pos,
                "Ref": ref,
                "Alt": alt,
                "MAF": maf,
                "rsid_field": fields[2],
            }
        )
    if not rows:
        return np.empty((0, len(sample_index)), dtype=np.int8), pd.DataFrame(marker_rows)
    genotype_matrix = np.vstack(rows)
    return genotype_matrix, pd.DataFrame(marker_rows)


def main() -> None:
    if GENOTYPE_DIR.exists() and any(GENOTYPE_DIR.iterdir()):
        raise FileExistsError(f"{GENOTYPE_DIR} already has content; refusing to overwrite.")
    GENOTYPE_DIR.mkdir(parents=True, exist_ok=True)

    eur_samples = load_eur_samples()
    print(f"Loaded {len(eur_samples)} European (EUR) sample IDs.")

    windows = build_window_list()
    print(f"Prepared {len(windows)} genomic windows "
          f"({len(CAUSAL_LOCI)} causal-locus windows + "
          f"{len(windows) - len(CAUSAL_LOCI)} background windows).")

    blocks_manifest = []
    causal_marker_records = []
    total_markers = 0
    start_time = time.time()

    # Sample-column index is resolved once per chromosome (VCF header order
    # is identical across per-chromosome files in this release, but we look
    # it up fresh per chromosome defensively rather than assuming that).
    sample_index_cache: dict[int, list[int]] = {}

    for window in windows:
        chrom = window["chrom"]
        if chrom not in sample_index_cache:
            url = VCF_BASE.format(chrom=chrom)
            tbx = pysam.TabixFile(url)
            header_cols = None
            for line in tbx.header:
                header_cols = line.split("\t")
            all_columns = header_cols[9:]
            column_position = {name: i for i, name in enumerate(all_columns)}
            sample_index_cache[chrom] = [column_position[s] for s in eur_samples]

        genotype_matrix, marker_df = fetch_window(
            chrom, window["start"], window["end"], eur_samples, sample_index_cache[chrom]
        )
        if genotype_matrix.shape[0] == 0:
            print(f"  {window['block_id']}: 0 markers passed filters, skipping block.")
            continue

        block_path = GENOTYPE_DIR / f"{window['block_id']}.npy"
        np.save(block_path, genotype_matrix.astype(np.int8))
        marker_df.insert(0, "block_id", window["block_id"])
        marker_map_path = GENOTYPE_DIR / f"{window['block_id']}_markers.csv"
        marker_df.to_csv(marker_map_path, index=False)

        total_markers += genotype_matrix.shape[0]
        blocks_manifest.append(
            {
                "block_id": window["block_id"],
                "chrom": chrom,
                "start": window["start"],
                "end": window["end"],
                "kind": window["kind"],
                "n_markers": int(genotype_matrix.shape[0]),
                "genotype_path": str(block_path.relative_to(OUT_DIR)),
                "marker_map_path": str(marker_map_path.relative_to(OUT_DIR)),
            }
        )
        print(f"  {window['block_id']}: {genotype_matrix.shape[0]:,} markers "
              f"(MAF>={MAF_MIN}), elapsed {time.time() - start_time:.0f}s")

        if window["kind"] == "causal_locus":
            causal_snp_name = window["causal_snp"]
            locus_def = next(l for l in CAUSAL_LOCI if l["name"] == causal_snp_name)
            hit = marker_df[marker_df["Pos"] == locus_def["pos"]]
            if hit.empty:
                raise RuntimeError(
                    f"Causal SNP {causal_snp_name} at chr{chrom}:{locus_def['pos']} "
                    f"was not found (or filtered out) in its own window; check MAF/positions."
                )
            row_index_in_block = marker_df.index[marker_df["Pos"] == locus_def["pos"]][0]
            causal_marker_records.append(
                {
                    "rsid": causal_snp_name,
                    "gene": locus_def["gene"],
                    "role": locus_def["role"],
                    "target_var_fraction": locus_def["target_var_fraction"],
                    "chrom": chrom,
                    "pos": locus_def["pos"],
                    "block_id": window["block_id"],
                    "row_index_in_block": int(row_index_in_block),
                    "maf_in_eur_sample": float(hit.iloc[0]["MAF"]),
                    "note": locus_def["note"],
                }
            )

    print(f"\nFetched {total_markers:,} markers across {len(blocks_manifest)} blocks "
          f"in {time.time() - start_time:.0f}s.")

    causal_df = pd.DataFrame(causal_marker_records)
    causal_df.to_csv(OUT_DIR / "causal_truth_POSTHOC_ONLY.csv", index=False)

    # --- Simulate the phenotype -------------------------------------------------
    rng = np.random.default_rng(RANDOM_SEED)
    n_samples = len(eur_samples)
    genetic_score = np.zeros(n_samples, dtype=np.float64)
    for record in causal_marker_records:
        block = np.load(GENOTYPE_DIR / f"{record['block_id']}.npy")
        dosage = block[record["row_index_in_block"]].astype(np.float64)
        standardized = (dosage - dosage.mean())
        sd = dosage.std()
        if sd > 0:
            standardized = standardized / sd
        weight = np.sqrt(record["target_var_fraction"])
        # Random sign per locus: direction of the simulated effect is
        # arbitrary for this recovery test (RF/GB importances do not depend
        # on effect sign), and no attempt is made here to encode real
        # allele-direction biology.
        sign = 1.0 if rng.random() < 0.5 else -1.0
        genetic_score += sign * weight * standardized

    genetic_score = genetic_score - genetic_score.mean()
    genetic_sd = genetic_score.std()
    if genetic_sd > 0:
        genetic_score = genetic_score / genetic_sd
    combined_genetic_var_fraction = 1.0 - TARGET_RESIDUAL_VAR_FRACTION
    noise = rng.normal(loc=0.0, scale=1.0, size=n_samples)  # unit-variance noise
    phenotype = (
        np.sqrt(combined_genetic_var_fraction) * genetic_score
        + np.sqrt(TARGET_RESIDUAL_VAR_FRACTION) * noise
    )
    # Final standardization keeps the simulated phenotype on a clean z-score
    # scale; only the *relative* variance partition across loci matters for
    # this recovery test, not the absolute scale of any real pigmentation
    # index.
    phenotype = (phenotype - phenotype.mean()) / phenotype.std()

    if not np.all(np.isfinite(phenotype)):
        raise RuntimeError("Non-finite values in simulated phenotype; aborting.")

    phenotype_df = pd.DataFrame({"sample": eur_samples, "pigmentation_score_simulated": phenotype})
    phenotype_df.to_csv(OUT_DIR / "phenotypes.csv", index=False)

    manifest = {
        "data_source": (
            "1000 Genomes Project 30x high-coverage call set (GRCh38), "
            "public FTP, no restricted access required"
        ),
        "sample_subset": "503 EUR-superpopulation samples (CEU/TSI/FIN/GBR/IBS)",
        "n_samples": n_samples,
        "n_blocks": len(blocks_manifest),
        "n_markers_total": total_markers,
        "maf_filter": MAF_MIN,
        "causal_loci": CAUSAL_LOCI,
        "target_residual_var_fraction": TARGET_RESIDUAL_VAR_FRACTION,
        "random_seed": RANDOM_SEED,
        "blocks": blocks_manifest,
    }
    (OUT_DIR / "simulation_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nWrote simulation_manifest.json, phenotypes.csv, causal_truth_POSTHOC_ONLY.csv.")
    print("Genetic variance fraction (combined):", round(combined_genetic_var_fraction, 3))
    print("Residual (noise) variance fraction:", round(TARGET_RESIDUAL_VAR_FRACTION, 3))


if __name__ == "__main__":
    main()
