# Run this script from the 02_arabidopsis_gwas folder:
#   Rscript run_ft16_gapit_targeted.R

suppressPackageStartupMessages(library(GAPIT))

Y <- read.csv("ft16_gapit_phenotype.csv", stringsAsFactors = FALSE)
Y$Taxa <- as.character(Y$Taxa)

kinship_table <- read.csv(
  "ft10_kinship.csv",
  row.names = 1,
  check.names = FALSE
)
kinship_matrix <- as.matrix(kinship_table)

# Restrict to accessions present in both FT16 phenotype and the kinship matrix
common_taxa <- intersect(Y$Taxa, rownames(kinship_matrix))
message("Accessions with phenotype + kinship: ", length(common_taxa))

Y <- Y[match(common_taxa, Y$Taxa), , drop = FALSE]
kinship_matrix <- kinship_matrix[
  match(common_taxa, rownames(kinship_matrix)),
  match(common_taxa, colnames(kinship_matrix))
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

# Only the 4 blocks containing the FT10 candidate loci (Chr1 FT, Chr5 ZTL region,
# Chr5 ~18.59Mb cluster, Chr3 rare-variant hit)
target_blocks <- c(
  "ft10_chr1_block022",
  "ft10_chr5_block021",
  "ft10_chr5_block017",
  "ft10_chr3_block019"
)

results_dir <- "ft16_gapit_results"
dir.create(results_dir, showWarnings = FALSE)

for (block_name in target_blocks) {
  block_file <- file.path("ft10_blocks", paste0(block_name, ".csv.gz"))
  marker_file <- file.path("ft10_blocks", paste0(block_name, "_markers.csv"))
  result_file <- file.path(results_dir, paste0(block_name, "_GWAS.csv"))

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
    memo = block_name
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

message("Targeted FT16 replication blocks completed.")
