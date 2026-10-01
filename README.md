# Artificial Intelligence in Genomics: Validating Machine Learning for Genetic Association

Analysis code for the Matura thesis *Artificial Intelligence in Genomics:
Validating Machine Learning for Genetic Association* 


The thesis compares classical genome-wide association studies (GWAS) with the
machine-learning methods Random Forest and Gradient Boosting. The workflow was
developed on *Arabidopsis thaliana* flowering-time data, evaluated on synthetic
datasets with planted causal markers, and finally applied to real human
genotypes from the 1000 Genomes Project combined with a simulated
pigmentation score.

This repository contains the code used for the analyses described in the
Materials and Methods chapter. It does not contain the input data (see
[Data](#data)) or the code used to write and format the thesis document.

## Contents

| Folder | Thesis section | Content |
| --- | --- | --- |
| [`01_tutorial_bean_gwas`](01_tutorial_bean_gwas) | 2.1 | R console history of the introductory GAPIT tutorial (bean disease resistance) |
| [`02_arabidopsis_gwas`](02_arabidopsis_gwas) | 2.2–2.3, 3.1 | Classical FT10 GWAS with GAPIT, the targeted FT16 analysis, and the initial thinned scan |
| [`03_arabidopsis_ml`](03_arabidopsis_ml) | 2.4, 3.2 | Exploratory Random Forest and Gradient Boosting analyses of FT10 |
| [`04_synthetic_benchmarks`](04_synthetic_benchmarks) | 2.5, 3.3–3.4 | Blind and context-calibrated simulations, multi-seed assessment, diagnostic plots |
| [`05_human_pilot`](05_human_pilot) | 2.6.1–2.6.3 | Simulated pigmentation score and 26-window human pilot analysis |
| [`06_human_genomewide`](06_human_genomewide) | 2.6.4, 3.5 | Genome-wide association scan and nested machine-learning analysis of 6,131,654 SNPs |

## Software

The analyses were run on macOS with the following software:

- **Python 3.9.6** with the packages listed in [`requirements.txt`](requirements.txt)
  (NumPy, pandas, SciPy, scikit-learn, Matplotlib, h5py, hdf5plugin, pysam).
- **R 4.6.1** with **GAPIT 4.1.0** for the classical GWAS analyses.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Rscript install_r_packages.R
```

## Data

The input data are not included in this repository. They are publicly
available, except for the tutorial files, which are course material.

| Data | Source | Where to place it |
| --- | --- | --- |
| Bean tutorial genotypes and phenotypes (`extBALSIT_hmp.txt`, `ALS_phenotypes.txt`) | Course exercise "GWAS Exercise 2: Bean Disease Resistance" (Molecular Plant Breeding lecture, FS25); not redistributed here | `01_tutorial_bean_gwas/` |
| *Arabidopsis* imputed SNP matrix and IBS kinship matrix (`imputed_snps_binary.hdf5`, `kinship_ibs_mac5.hdf5`) | 1001 Genomes Project: download `1001_SNP_MATRIX.tar.gz` from <https://1001genomes.org/data/GMI-MPI/releases/v3.1/SNP_matrix_imputed_hdf5/> | extract so that the files are in `02_arabidopsis_gwas/1001_SNP_MATRIX/` |
| FT10 flowering time (AraPheno phenotype 261) | <https://arapheno.1001genomes.org/rest/phenotype/261/values.csv> | save as `02_arabidopsis_gwas/FT10_Phenotype.csv` |
| FT16 flowering time (AraPheno phenotype 262) | <https://arapheno.1001genomes.org/rest/phenotype/262/values.csv> | save as `02_arabidopsis_gwas/FT16_Phenotype.csv` |
| 1000 Genomes sample panel | <https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel> | save as `05_human_pilot/1000genomes_sample_panel.tsv` |
| 1000 Genomes high-coverage genotypes (GRCh38) | Read directly from the public 1000 Genomes server by the scripts (internet connection required) | not needed locally |

Both flowering-time phenotypes belong to the AraPheno study "1001genomes
flowering time phenotypes".

Example download commands:

```bash
curl -L -o 02_arabidopsis_gwas/FT10_Phenotype.csv https://arapheno.1001genomes.org/rest/phenotype/261/values.csv
curl -L -o 02_arabidopsis_gwas/FT16_Phenotype.csv https://arapheno.1001genomes.org/rest/phenotype/262/values.csv
curl -L -o 05_human_pilot/1000genomes_sample_panel.tsv https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel
```

## Running the analyses

The steps build on each other and should be run in the order below. Each
script writes its outputs into its own folder.

### 1. Tutorial (`01_tutorial_bean_gwas`)

The tutorial was carried out interactively in the R console. See
[`01_tutorial_bean_gwas/README.md`](01_tutorial_bean_gwas/README.md).

### 2. Classical *Arabidopsis* GWAS (`02_arabidopsis_gwas`)

Run from inside the folder:

```bash
cd 02_arabidopsis_gwas

# Initial scan on every 20th SNP (superseded development run, Section 2.3.1)
(cd initial_thinned_scan && python extract_ft10_gwas.py && Rscript run_ft10_gapit_thinned.R)

# Full-density FT10 scan (MLM with kinship matrix and three PCs)
python prepare_ft10_blocks.py        # MAF filter and 108 genotype blocks
Rscript run_ft10_gapit_blocks.R      # GAPIT MLM for every block
python postprocess_ft10_gwas.py      # merge blocks, significance summary
Rscript plot_ft10_gwas.R             # Manhattan and QQ plots
python plot_candidate_regions.py     # zoomed plots of candidate regions

# Targeted FT16 analysis of four candidate blocks
python prepare_ft16_phenotype.py
Rscript run_ft16_gapit_targeted.R
python plot_ft16_replication.py
```

`run_ft16_candidate_blocks.R` is an earlier two-block version of the FT16
analysis that was superseded by `run_ft16_gapit_targeted.R`.

### 3. Machine learning on FT10 (`03_arabidopsis_ml`)

```bash
cd 03_arabidopsis_ml/exploratory_every_10th_snp
python build_ml_matrix.py && python run_ml_analysis.py

cd ../full_resolution
python build_ml_matrix_full.py && python run_ml_analysis.py
```

Both analyses select candidate markers before cross-validation. Their
prediction values are therefore exploratory and may be optimistic
(Section 2.4.2 of the thesis).

### 4. Synthetic benchmarks (`04_synthetic_benchmarks`)

```bash
cd 04_synthetic_benchmarks
python run_synthetic_ft10_ml.py --scenario blind
python run_synthetic_ft10_ml.py --scenario context_calibrated
python multi_seed_validation/run_multi_seed_validation.py
python multi_seed_validation/make_multi_seed_plots.py
```

`make_ai_run_plots.py` creates the diagnostic plots for the two simulations
and the human pilot; run it after step 5.

### 5. Human pilot (`05_human_pilot`)

```bash
cd 05_human_pilot
python fetch_and_simulate_human_pigmentation.py   # genotype windows and simulated score
python analyze_human_pigmentation.py              # nested cross-validation
```

### 6. Genome-wide human analysis (`06_human_genomewide`)

```bash
cd 06_human_genomewide
python run_genomewide.py --stage all
```

The script reuses the simulated score and the sample panel from
`05_human_pilot`. `--stage extract` and `--stage analyze` run the two parts
separately; completed chromosomes are saved and skipped when the extraction
is restarted.

## Computational requirements

Several steps are demanding. The full-resolution *Arabidopsis* analyses were
run with 17 GB of RAM, the synthetic benchmarks write several gigabytes of
intermediate genotype files per run, and the genome-wide human analysis
downloads and processes genotypes for all 22 autosomes. Complete runs take
several hours.

## Reproducibility

- All random steps use fixed seeds that are set in the scripts, for example
  20260928 and 20260929 for the two main simulations, 20270101 as the base
  seed of the multi-seed runs, 20260401 for the simulated pigmentation score,
  and 20260929 for the genome-wide cross-validation folds.
- `analyze_human_pigmentation.py` derives some model seeds from Python's
  built-in `hash()` of the window names. Because Python randomizes string
  hashing between sessions, pilot results can differ slightly between runs
  unless `PYTHONHASHSEED` is fixed. The genome-wide analysis does not use
  this mechanism.
- In the genome-wide analysis, the principal components are calculated once
  from all 503 individuals (without phenotype values) before
  cross-validation, as described in the thesis.

## Notes on this repository

- Absolute file paths from the original project folders were replaced by
  paths relative to the repository. Some comments, descriptive labels, and
  printed messages were revised for accuracy. Model settings, parameters,
  seeds, and computations are unchanged.
- The tutorial (`01_tutorial_bean_gwas`) is included as the unedited R console
  history, because it was run interactively.
- `02_arabidopsis_gwas/initial_thinned_scan/run_ft10_gapit_thinned.R` contains
  the commands of the initial scan, transcribed from the saved R console log.
- `02_arabidopsis_gwas/prepare_ft16_phenotype.py` reconstructs a preparation
  step that was originally run as a one-off command. It reproduces the
  original FT16 input file exactly.
- As declared in the thesis, a generative-AI coding assistant was used under
  the author's direction to help write and revise these scripts.

## Data and software references

- 1000 Genomes Project Consortium. (2015). A global reference for human genetic variation. *Nature, 526*(7571), 68–74. https://doi.org/10.1038/nature15393
- 1001 Genomes Consortium. (2016). 1,135 genomes reveal the global pattern of polymorphism in *Arabidopsis thaliana*. *Cell, 166*(2), 481–491. https://doi.org/10.1016/j.cell.2016.05.063
- Byrska-Bishop, M., et al. (2022). High-coverage whole-genome sequencing of the expanded 1000 Genomes Project cohort including 602 trios. *Cell, 185*(18), 3426–3440.e19. https://doi.org/10.1016/j.cell.2022.08.004
- Lipka, A. E., et al. (2012). GAPIT: Genome association and prediction integrated tool. *Bioinformatics, 28*(18), 2397–2399. https://doi.org/10.1093/bioinformatics/bts444
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Seren, Ü., et al. (2017). AraPheno: A public database for *Arabidopsis thaliana* phenotypes. *Nucleic Acids Research, 45*(D1), D1054–D1059. https://doi.org/10.1093/nar/gkw986
- Togninalli, M., et al. (2020). AraPheno and the AraGWAS Catalog 2020. *Nucleic Acids Research, 48*(D1), D1063–D1068. https://doi.org/10.1093/nar/gkz925
- Wang, J., & Zhang, Z. (2021). GAPIT version 3: Boosting power and accuracy for genomic association and prediction. *Genomics, Proteomics & Bioinformatics, 19*(4), 629–640. https://doi.org/10.1016/j.gpb.2021.08.005
