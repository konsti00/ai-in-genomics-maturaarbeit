# Initial FT10 GWAS on every 20th SNP (superseded development run).
#
# This analysis was originally run interactively in the R console. The
# commands below are transcribed from the saved console log; only the file
# location handling was adapted for this repository.
#
# Inputs (created by extract_ft10_gwas.py in this folder):
#   ft10_genotypes.csv.gz, ft10_marker_map.csv, ft10_gapit_phenotype.csv
#
# Run this script from the initial_thinned_scan folder:
#   Rscript run_ft10_gapit_thinned.R

raw_genotypes <- read.csv(
  gzfile("ft10_genotypes.csv.gz"),
  check.names = FALSE
)

marker_map <- read.csv(
  "ft10_marker_map.csv",
  stringsAsFactors = FALSE
)

Y <- read.csv(
  "ft10_gapit_phenotype.csv",
  stringsAsFactors = FALSE
)

GD <- data.frame(
  Taxa = colnames(raw_genotypes)[-1],
  t(raw_genotypes[, -1, drop = FALSE]),
  check.names = FALSE
)

GM <- marker_map[, c("Marker", "Chr", "Pos")]
common <- intersect(Y$Taxa, GD$Taxa)

Y <- Y[match(common, Y$Taxa), , drop = FALSE]
GD <- GD[match(common, GD$Taxa), , drop = FALSE]

stopifnot(identical(
  as.character(Y$Taxa),
  as.character(GD$Taxa)
))

library(GAPIT)

gwas_result <- GAPIT(
  Y = Y,
  GD = GD,
  GM = GM,
  model = "MLM",
  PCA.total = 3,
  file.output = TRUE
)
