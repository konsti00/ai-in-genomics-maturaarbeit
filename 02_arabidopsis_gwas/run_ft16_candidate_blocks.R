# Run this script from the 02_arabidopsis_gwas folder:
#   Rscript run_ft16_candidate_blocks.R

suppressPackageStartupMessages(library(GAPIT))

Y <- read.csv("ft16_gapit_phenotype.csv", stringsAsFactors = FALSE)
Y$Taxa <- as.character(Y$Taxa)

kinship_table <- read.csv(
  "ft10_kinship.csv",
  row.names = 1,
  check.names = FALSE
)

# Keep only FT16 accessions that are actually present in the kinship matrix,
# so Y, CV, and GD all stay perfectly aligned (no NA rows from a failed match).
Y <- Y[Y$Taxa %in% rownames(kinship_table), , drop = FALSE]

kinship_matrix <- as.matrix(kinship_table)
kinship_matrix <- kinship_matrix[
  match(Y$Taxa, rownames(kinship_matrix)),
  match(Y$Taxa, colnames(kinship_matrix))
]
stopifnot(all(rownames(kinship_matrix) == Y$Taxa))
stopifnot(all(colnames(kinship_matrix) == Y$Taxa))

kinship <- data.frame(
  taxa = Y$Taxa,
  kinship_matrix,
  check.names = FALSE
)

kinship_eigen <- eigen(kinship_matrix, symmetric = TRUE)
positive <- kinship_eigen$values > 1e-8
n_pcs <- min(3, sum(positive))
CV <- data.frame(
  Taxa = as.character(Y$Taxa),
  kinship_eigen$vectors[, positive, drop = FALSE][, seq_len(n_pcs), drop = FALSE],
  check.names = FALSE
)
names(CV)[-1] <- paste0("PC", seq_len(n_pcs))

# Only the two blocks covering the FT10 candidate loci:
# chr1_block022 -> FT (AT1G65480) region
# chr5_block021 -> ZTL (AT5G57360) / broad chr5 candidate region
block_files <- file.path(
  "ft10_blocks",
  c("ft10_chr1_block022.csv.gz", "ft10_chr5_block021.csv.gz")
)
stopifnot(all(file.exists(block_files)))

results_dir <- "ft16_gapit_results"
dir.create(results_dir, showWarnings = FALSE)

for (block_file in block_files) {
  block_name <- sub("\\.csv\\.gz$", "", basename(block_file))
  result_file <- file.path(results_dir, paste0(block_name, "_FT16_GWAS.csv"))

  if (file.exists(result_file)) {
    message("Skipping completed block: ", block_name)
    next
  }

  message("Loading ", block_name)
  raw <- read.csv(
    gzfile(block_file),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  marker_file <- file.path("ft10_blocks", paste0(block_name, "_markers.csv"))
  GM <- read.csv(marker_file, stringsAsFactors = FALSE)

  GD <- data.frame(
    Taxa = as.character(colnames(raw)[-1]),
    t(raw[, -1, drop = FALSE]),
    check.names = FALSE
  )
  GD <- GD[match(Y$Taxa, GD$Taxa), , drop = FALSE]
  stopifnot(identical(as.character(GD$Taxa), as.character(Y$Taxa)))

  block_result <- GAPIT(
    Y = Y,
    GD = GD,
    GM = GM,
    KI = kinship,
    CV = CV,
    PCA.total = 0,
    model = "MLM",
    file.output = FALSE,
    memo = paste0(block_name, "_FT16")
  )

  write.csv(
    block_result$GWAS,
    result_file,
    row.names = FALSE
  )

  rm(raw, GD, GM, block_result)
  gc(verbose = FALSE)
  message("Saved ", result_file)
}

message("FT16 candidate-region blocks completed.")
