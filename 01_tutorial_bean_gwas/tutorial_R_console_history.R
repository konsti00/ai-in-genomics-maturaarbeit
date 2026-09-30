install.packages(c(#
  "data.table",#
  "tidyverse",#
  "vcfR",#
  "qqman",#
  "ggplot2"#
))
if (!requireNamespace("BiocManager", quietly = TRUE)) {#
  install.packages("BiocManager")#
}#
#
BiocManager::install(c(#
  "SNPRelate",#
  "GWASTools",#
  "gdsfmt"#
))
install.packages("remotes")#
remotes::install_github("jiabowang/GAPIT")
getwd()
setwd()/Users/konstantin/Maturaarbeit
setwd("/Users/konstantin/Maturaarbeit")
getwd()
library(GAPIT)
genotype <- read.delim(#
  "extBALSIT_hmp.txt",#
  header = TRUE,#
  check.names = FALSE,#
  na.strings = "N"#
)#
#
phenotype <- read.delim(#
  "ALS_phenotypes.txt",#
  header = TRUE,#
  check.names = FALSE,#
  na.strings = "NA"#
)
dim(genotype)#
dim(phenotype)#
#
head(genotype[, 1:15])#
head(phenotype)#
#
all(phenotype$taxa %in% colnames(genotype))
marker_data <- genotype[, -(1:11)]#
#
marker_missing <- rowMeans(is.na(marker_data))#
sample_missing <- colMeans(is.na(marker_data))#
#
keep_markers <- marker_missing <= 0.20#
keep_samples <- sample_missing <= 0.20#
#
genotype_qc <- genotype[keep_markers, c(TRUE, keep_samples)]
nrow(genotype_qc)#
ncol(genotype_qc) - 11
genotype_ids <- colnames(genotype_qc)[-(1:11)]#
#
phenotype_qc <- phenotype[#
  phenotype$taxa %in% genotype_ids,#
]#
#
phenotype_qc <- phenotype_qc[#
  match(genotype_ids, phenotype_qc$taxa),#
]#
#
stopifnot(identical(phenotype_qc$taxa, genotype_ids))
darien <- phenotype_qc[, c("taxa", "Darien")]#
darien <- darien[!is.na(darien$Darien), ]#
#
genotype_darien <- genotype_qc[#
  , c(TRUE, colnames(genotype_qc)[-(1:11)] %in% darien$taxa)#
]#
#
genotype_darien <- genotype_darien[#
  , c(TRUE, match(darien$taxa, colnames(genotype_darien)[-(1:11)]) + 11)#
]
metadata_columns <- seq_len(11)#
sample_columns <- 12:ncol(genotype)#
#
keep_samples <- colMeans(is.na(genotype[, sample_columns])) <= 0.20#
keep_markers <- rowMeans(is.na(genotype[, sample_columns])) <= 0.20#
#
genotype_qc <- genotype[#
  keep_markers,#
  c(metadata_columns, sample_columns[keep_samples])#
]
darien <- phenotype_qc[, c("taxa", "Darien")]#
darien <- darien[!is.na(darien$Darien), ]#
#
genotype_ids <- colnames(genotype_qc)[-(1:11)]#
#
missing_ids <- setdiff(darien$taxa, genotype_ids)#
#
if (length(missing_ids) > 0) {#
  stop(#
    "These phenotype IDs are missing from the genotype data: ",#
    paste(missing_ids, collapse = ", ")#
  )#
}#
#
darien <- darien[match(genotype_ids, darien$taxa), ]#
darien <- darien[!is.na(darien$Darien), ]#
#
genotype_darien <- genotype_qc[#
  ,#
  c(colnames(genotype_qc)[1:11], darien$taxa),#
  drop = FALSE#
]
genotype_ids <- colnames(genotype_qc)[-(1:11)]#
#
darien <- phenotype_qc[, c("taxa", "Darien")]#
darien <- darien[!is.na(darien$Darien), ]#
#
darien <- darien[darien$taxa %in% genotype_ids, ]#
#
# Put phenotype rows in exactly the same order as genotype columns#
darien <- darien[match(genotype_ids, darien$taxa), ]#
darien <- darien[!is.na(darien$Darien), ]#
#
# Re-select genotype columns to match the remaining phenotype rows#
genotype_darien <- genotype_qc[#
  ,#
  c(colnames(genotype_qc)[1:11], darien$taxa),#
  drop = FALSE#
]
identical(#
  colnames(genotype_darien)[-(1:11)],#
  darien$taxa#
)
col63 <- phenotype_qc[, c("taxa", "COL_63-63")]
result_darien <- GAPIT(#
  Y = darien,#
  G = genotype_darien,#
  model = "MLM",#
  PCA.total = 3,#
  SNP.MAF = 0.05#
)
traceback()
length(genotype_ids)#
nrow(darien)#
length(darien$taxa)#
length(darien$Darien)
genotype_ids <- colnames(genotype_qc)[-(1:11)]#
#
darien <- phenotype_qc[, c("taxa", "Darien")]#
darien <- darien[#
  !is.na(darien$Darien) &#
  darien$taxa %in% genotype_ids,#
  ,#
  drop = FALSE#
]#
#
cat("Matching Darien samples:", nrow(darien), "\n")
sample_missingness <- colMeans(#
  is.na(genotype[, sample_columns]),#
  na.rm = FALSE#
)#
#
missingness_table <- data.frame(#
  taxa = names(sample_missingness),#
  missingness = as.numeric(sample_missingness),#
  row.names = NULL#
)
nrow(darien)#
ncol(genotype_darien)#
length(darien$taxa)#
length(colnames(genotype_darien))
genotype_sample_ids <- colnames(genotype_darien)[-(1:11)]#
#
identical(genotype_sample_ids, darien$taxa)
stopifnot(#
  nrow(darien) == length(genotype_sample_ids),#
  identical(genotype_sample_ids, darien$taxa)#
)
result_darien <- GAPIT(#
  Y = darien[, c("taxa", "Darien")],#
  G = genotype_darien,#
  model = "MLM",#
  PCA.total = 3,#
  SNP.MAF = 0.05#
)
# Check exact sample alignment#
genotype_sample_ids <- colnames(genotype_darien)[-(1:11)]#
#
identical(genotype_sample_ids, darien$taxa)#
anyDuplicated(genotype_sample_ids)#
anyDuplicated(darien$taxa)#
#
# Check genotype dimensions and contents#
dim(genotype_darien)#
table(unlist(genotype_darien[, -(1:11)]), useNA = "always")#
#
# Check marker metadata#
head(genotype_darien[, 1:11])#
sum(is.na(genotype_darien$chrom))#
sum(is.na(genotype_darien$pos))
Y_darien <- darien[, c("taxa", "Darien")]#
#
result_darien <- GAPIT(#
  Y = Y_darien,#
  G = genotype_darien,#
  model = "GLM",#
  PCA.total = 0,#
  SNP.MAF = 0#
)
gmat <- as.matrix(genotype_darien[, -(1:11), drop = FALSE])#
#
missing <- is.na(gmat) | gmat == "N"#
#
number_of_alleles <- apply(#
  gmat,#
  1,#
  function(x) {#
    observed <- x[!is.na(x) & x != "N"]#
    length(unique(observed))#
  }#
)#
#
table(number_of_alleles)#
#
keep_polymorphic <- number_of_alleles >= 2#
#
genotype_darien_poly <- genotype_darien[#
  keep_polymorphic,#
  ,#
  drop = FALSE#
]#
#
dim(genotype_darien_poly)
genotype_darien_poly <- genotype_darien_poly[#
  !duplicated(genotype_darien_poly$rs.),#
  ,#
  drop = FALSE#
]
identical(#
  colnames(genotype_darien_poly)[-(1:11)],#
  Y_darien$taxa#
)
result_darien <- GAPIT(#
  Y = Y_darien,#
  G = genotype_darien_poly,#
  model = "GLM",#
  PCA.total = 0,#
  SNP.MAF = 0.01#
)
Y_darien <- data.frame(#
  taxa = as.character(darien$taxa),#
  Darien = as.numeric(darien$Darien),#
  check.names = FALSE#
)#
#
result_darien <- GAPIT(#
  Y = Y_darien,#
  G = genotype_darien_poly,#
  model = "GLM",#
  PCA.total = 3,#
  SNP.MAF = 0.01#
)
genotype_darien_poly[, -(1:11)] <- lapply(#
  genotype_darien_poly[, -(1:11), drop = FALSE],#
  function(x) {#
    x[is.na(x) | x == ""] <- "N"#
    as.character(x)#
  }#
)
Y_darien <- data.frame(#
  taxa = as.character(darien$taxa),#
  Darien = as.numeric(darien$Darien),#
  check.names = FALSE#
)#
#
result_darien <- GAPIT(#
  Y = Y_darien,#
  G = genotype_darien_poly,#
  model = "GLM",#
  PCA.total = 3,#
  SNP.MAF = 0.01#
)
traceback()
sum(is.na(Y_darien$Darien))#
sd(Y_darien$Darien)#
sum(is.na(genotype_darien_poly))#
nrow(genotype_darien_poly)#
ncol(genotype_darien_poly)
# Genotype sample IDs#
genotype_ids <- colnames(genotype_darien_poly)[-(1:11)]#
#
# Make GAPIT's expected phenotype column name#
Y_darien <- data.frame(#
  Taxa = as.character(darien$taxa),#
  Darien = as.numeric(darien$Darien),#
  check.names = FALSE#
)#
#
# Keep only exact shared IDs#
common_ids <- intersect(Y_darien$Taxa, genotype_ids)#
#
length(common_ids)#
length(setdiff(Y_darien$Taxa, genotype_ids))#
length(setdiff(genotype_ids, Y_darien$Taxa))
Y_darien <- Y_darien[#
  match(common_ids, Y_darien$Taxa),#
  ,#
  drop = FALSE#
]#
#
genotype_darien_poly <- genotype_darien_poly[#
  ,#
  c(#
    colnames(genotype_darien_poly)[1:11],#
    common_ids#
  ),#
  drop = FALSE#
]#
#
stopifnot(#
  identical(#
    Y_darien$Taxa,#
    colnames(genotype_darien_poly)[-(1:11)]#
  )#
)
table(Y_darien$Taxa %in% colnames(genotype_darien_poly))#
table(colnames(genotype_darien_poly)[-(1:11)] %in% Y_darien$Taxa)
result_darien <- GAPIT(#
  Y = Y_darien,#
  G = genotype_darien_poly,#
  model = "GLM",#
  PCA.total = 3,#
  SNP.MAF = 0.01#
)
G <- genotype_darien_poly#
#
# Genotype calls: markers are rows, taxa are columns#
calls <- as.matrix(G[, -(1:11), drop = FALSE])#
#
# Preserve missing values#
calls[calls == "N" | calls == ""] <- NA#
#
# Encode alleles as 0 and 2#
encoded <- matrix(#
  NA_real_,#
  nrow = nrow(calls),#
  ncol = ncol(calls),#
  dimnames = dimnames(calls)#
)#
#
for (i in seq_len(nrow(calls))) {#
  alleles <- strsplit(as.character(G$alleles[i]), "/", fixed = TRUE)[[1]]#
#
  encoded[i, calls[i, ] == alleles[1]] <- 0#
  encoded[i, calls[i, ] == alleles[2]] <- 2#
}#
#
# GAPIT GD format: taxa are rows, markers are columns#
GD <- data.frame(#
  Taxa = colnames(encoded),#
  t(encoded),#
  check.names = FALSE#
)#
#
# GAPIT GM format#
GM <- data.frame(#
  SNP = as.character(G$rs.),#
  Chromosome = as.numeric(G$chrom),#
  Position = as.numeric(G$pos),#
  stringsAsFactors = FALSE#
)#
#
# Remove duplicate or missing marker IDs#
keep <- !is.na(GM$SNP) &#
        GM$SNP != "" &#
        !duplicated(GM$SNP)#
#
GD <- GD[, c(TRUE, keep), drop = FALSE]#
GM <- GM[keep, , drop = FALSE]#
#
# Ensure phenotype and genotype taxa match exactly#
common_ids <- intersect(Y_darien$Taxa, GD$Taxa)#
#
Y_input <- Y_darien[#
  match(common_ids, Y_darien$Taxa),#
  ,#
  drop = FALSE#
]#
#
GD <- GD[#
  match(common_ids, GD$Taxa),#
  ,#
  drop = FALSE#
]#
#
stopifnot(identical(Y_input$Taxa, GD$Taxa))
dim(GD)#
dim(GM)#
table(unlist(GD[, -1]), useNA = "always")
result_darien <- GAPIT(#
  Y = Y_input,#
  GD = GD,#
  GM = GM,#
  model = "GLM",#
  PCA.total = 3,#
  SNP.MAF = 0.01#
)
# Numeric genotype matrix: taxa rows, markers columns#
numeric_genotypes <- as.matrix(GD[, -1, drop = FALSE])#
storage.mode(numeric_genotypes) <- "numeric"#
#
# Remove markers that are entirely missing#
keep <- colSums(!is.na(numeric_genotypes)) > 0#
numeric_genotypes <- numeric_genotypes[, keep, drop = FALSE]#
GM <- GM[keep, , drop = FALSE]#
#
# Mean-impute missing genotype values marker by marker#
for (j in seq_len(ncol(numeric_genotypes))) {#
  marker_mean <- mean(numeric_genotypes[, j], na.rm = TRUE)#
#
  if (is.finite(marker_mean)) {#
    numeric_genotypes[is.na(numeric_genotypes[, j]), j] <- marker_mean#
  }#
}#
#
GD <- data.frame(#
  Taxa = GD$Taxa,#
  numeric_genotypes,#
  check.names = FALSE#
)
sum(is.na(GD[, -1]))
all(vapply(GD[, -1], is.numeric, logical(1)))
result_darien <- GAPIT(#
  Y = Y_input,#
  GD = GD,#
  GM = GM,#
  model = "GLM",#
  PCA.total = 3,#
  SNP.MAF = 0.01#
)
getwd()
setwd("/Users/konstantin/MA 1. GWAS MLM ")
setwd("/Users/konstantin/MA 1. GWAS MLM")
result_darien_mlm <- GAPIT(#
  Y = Y_input,#
  GD = GD,#
  GM = GM,#
  model = "MLM",#
  PCA.total = 3,#
  SNP.MAF = 0.01#
)
lambda_gc <- function(p) {#
  median(qchisq(1 - p, df = 1), na.rm = TRUE) /#
    qchisq(0.5, df = 1)#
}
lambda_glm <- lambda_gc(glm_results$P.value)#
lambda_mlm <- lambda_gc(mlm_results$P.value)#
#
lambda_glm#
lambda_mlm
result_darien_glm <- GAPIT(#
  Y = Y_input,#
  GD = GD,#
  GM = GM,#
  model = "GLM",#
  PCA.total = 3,#
  SNP.MAF = 0.01#
)#
#
result_darien_mlm <- GAPIT(#
  Y = Y_input,#
  GD = GD,#
  GM = GM,#
  model = "MLM",#
  PCA.total = 3,#
  SNP.MAF = 0.01#
)#
#
glm_results <- result_darien_glm$GWAS#
mlm_results <- result_darien_mlm$GWAS
names(result_darien_glm)#
names(result_darien_mlm)#
names(glm_results)#
head(glm_results)
lambda_gc <- function(p) {#
  p <- p[is.finite(p) & p > 0 & p <= 1]#
  median(qchisq(1 - p, df = 1)) / qchisq(0.5, df = 1)#
}#
#
lambda_glm <- lambda_gc(glm_results$P.value)#
lambda_mlm <- lambda_gc(mlm_results$P.value)#
#
lambda_glm#
lambda_mlm
glm_results[["Position "]]
lambda_glm <- lambda_gc(glm_results$P.value)#
lambda_mlm <- lambda_gc(mlm_results$P.value)#
#
data.frame(#
  model = c("GLM", "MLM"),#
  lambda = c(lambda_glm, lambda_mlm)#
)
qqplot(#
  glm_results$P.value,#
  mlm_results$P.value,#
  log = "xy",#
  xlab = "GLM p-values",#
  ylab = "MLM p-values"#
)#
abline(0, 1, col = "red")
alpha_bonferroni <- 0.05 / nrow(glm_results)#
#
glm_hits <- glm_results[#
  glm_results$P.value < alpha_bonferroni, ]#
#
mlm_hits <- mlm_results[#
  mlm_results$P.value < alpha_bonferroni, ]#
#
nrow(glm_hits)#
nrow(mlm_hits)#
#
intersect(glm_hits$SNP, mlm_hits$SNP)
common_snps <- intersect(glm_hits$SNP, mlm_hits$SNP)#
#
merge(#
  glm_results[glm_results$SNP %in% common_snps,#
              c("SNP", "Chromosome", "Position", "P.value", "effect")],#
  mlm_results[mlm_results$SNP %in% common_snps,#
              c("SNP", "P.value", "effect")],#
  by = "SNP",#
  suffixes = c("_GLM", "_MLM")#
)
candidate_region <- GM[#
  GM$Chromosome == 8 &#
    GM$Position >= 60999071 &#
    GM$Position <= 63279357,#
  ,#
  drop = FALSE#
]#
#
nrow(candidate_region)#
head(candidate_region)
candidate_region <- GM[#
+   GM$Chromosome == 8 &#
+     GM$Position >= 59999071 &#
+     GM$Position <= 64279357,#
+   ,#
+   drop = FALSE#
+ ]#
> #
> nrow(candidate_region)#
[1] 232#
> head(candidate_region)
candidate_region <- GM[#
  GM$Chromosome == 8 &#
    GM$Position >= 59999071 &#
    GM$Position <= 64279357,#
  ,#
  drop = FALSE#
]#
#
nrow(candidate_region)#
head(candidate_region)
region_snps <- candidate_region$SNP#
#
region_genotypes <- GD[#
  ,#
  c("Taxa", region_snps),#
  drop = FALSE#
]
ncol(region_genotypes)
region_genotypes <- GD[#
  ,#
  c("Taxa", region_snps),#
  drop = FALSE#
]
colnames(region_genotypes)[1:5]#
colnames(region_genotypes)[229:233]#
dim(region_genotypes)
region_snps <- as.character(candidate_region$SNP)#
#
sum(region_snps %in% colnames(GD))#
head(setdiff(region_snps, colnames(GD)))#
head(colnames(GD))
# Genotype calls: markers in rows, taxa in columns#
calls <- as.matrix(genotype_darien_poly[, -(1:11), drop = FALSE])#
#
calls[calls == "N" | calls == ""] <- NA#
#
# Convert calls to numeric dosage#
dosage <- matrix(#
  NA_real_,#
  nrow = nrow(calls),#
  ncol = ncol(calls),#
  dimnames = list(genotype_darien_poly$rs., colnames(calls))#
)#
#
for (i in seq_len(nrow(calls))) {#
  alleles <- strsplit(#
    as.character(genotype_darien_poly$alleles[i]),#
    "/",#
    fixed = TRUE#
  )[[1]]#
#
  dosage[i, calls[i, ] == alleles[1]] <- 0#
  dosage[i, calls[i, ] == alleles[2]] <- 2#
}#
#
# Transpose: taxa become rows, SNPs become columns#
GD <- data.frame(#
  Taxa = colnames(calls),#
  t(dosage),#
  check.names = FALSE#
)#
#
# Explicitly restore marker names#
colnames(GD) <- c("Taxa", rownames(dosage))
sum(region_snps %in% colnames(GD))
region_genotypes <- GD[#
  ,#
  c("Taxa", region_snps),#
  drop = FALSE#
]
region_genotypes <- GD[#
  ,#
  c("Taxa", region_snps_found),#
  drop = FALSE#
]
region_snps <- as.character(candidate_region$SNP)#
#
region_snps_found <- intersect(#
  region_snps,#
  colnames(GD)#
)#
#
length(region_snps_found)
region_genotypes <- GD[#
  ,#
  c("Taxa", region_snps_found),#
  drop = FALSE#
]
region_genotypes <- GD[#
  ,#
  c("Taxa", region_snps_found),#
  drop = FALSE#
]
ncol(region_genotypes) - 1
# Remove the Taxa column and ensure numeric genotype values#
region_matrix <- as.matrix(#
  region_genotypes[, -1, drop = FALSE]#
)#
#
storage.mode(region_matrix) <- "numeric"#
#
# Calculate r²#
ld_r2 <- cor(#
  region_matrix,#
  use = "pairwise.complete.obs"#
)^2#
#
rownames(ld_r2) <- colnames(region_matrix)#
colnames(ld_r2) <- colnames(region_matrix)
marker_sd <- apply(#
  region_matrix,#
  2,#
  sd,#
  na.rm = TRUE#
)#
#
zero_variance <- names(marker_sd)[#
  !is.finite(marker_sd) | marker_sd == 0#
]#
#
length(zero_variance)#
zero_variance
keep_ld_markers <- is.finite(marker_sd) & marker_sd > 0#
#
region_matrix_ld <- region_matrix[#
  ,#
  keep_ld_markers,#
  drop = FALSE#
]#
#
ld_r2 <- cor(#
  region_matrix_ld,#
  use = "pairwise.complete.obs"#
)^2#
#
rownames(ld_r2) <- colnames(region_matrix_ld)#
colnames(ld_r2) <- colnames(region_matrix_ld)
sum(is.na(region_matrix))#
sum(!is.finite(region_matrix))
complete_counts <- crossprod(#
  !is.na(region_matrix)#
)#
#
summary(as.vector(complete_counts))
region_matrix_ld <- region_matrix#
#
# Convert non-finite values to NA#
region_matrix_ld[!is.finite(region_matrix_ld)] <- NA#
#
# Mean-impute each marker#
for (j in seq_len(ncol(region_matrix_ld))) {#
  marker_mean <- mean(region_matrix_ld[, j], na.rm = TRUE)#
#
  if (!is.finite(marker_mean)) {#
    region_matrix_ld[, j] <- NULL#
  } else {#
    region_matrix_ld[is.na(region_matrix_ld[, j]), j] <- marker_mean#
  }#
}#
#
# Remove any remaining zero-variance markers#
marker_sd <- apply(region_matrix_ld, 2, sd)#
#
region_matrix_ld <- region_matrix_ld[#
  ,#
  is.finite(marker_sd) & marker_sd > 0,#
  drop = FALSE#
]#
#
# Calculate LD#
ld_r2 <- cor(region_matrix_ld)^2#
#
rownames(ld_r2) <- colnames(region_matrix_ld)#
colnames(ld_r2) <- colnames(region_matrix_ld)
anyNA(ld_r2)#
any(!is.finite(ld_r2))
shared_snps_present <- intersect(#
  shared_snps,#
  colnames(ld_r2)#
)#
#
ld_r2[#
  shared_snps_present,#
  shared_snps_present,#
  drop = FALSE#
]
install.packages("pheatmap")#
library(pheatmap)#
#
pheatmap(#
  ld_r2,#
  cluster_rows = FALSE,#
  cluster_cols = FALSE,#
  show_rownames = FALSE,#
  show_colnames = FALSE,#
  main = "LD around chromosome 8 candidate region"#
)
hit_labels <- ifelse(#
  colnames(ld_r2) %in% shared_snps,#
  colnames(ld_r2),#
  ""#
)#
#
pheatmap(#
  ld_r2,#
  cluster_rows = FALSE,#
  cluster_cols = FALSE,#
  labels_row = hit_labels,#
  labels_col = hit_labels,#
  main = "Chromosome 8 LD and shared GWAS hits"#
)
shared_snps <- intersect(#
  glm_hits$SNP,#
  mlm_hits$SNP#
)#
#
shared_snps
hit_labels <- ifelse(#
  colnames(ld_r2) %in% shared_snps,#
  colnames(ld_r2),#
  ""#
)
shared_snps[shared_snps %in% colnames(ld_r2)]
pheatmap::pheatmap(#
  ld_r2,#
  cluster_rows = FALSE,#
  cluster_cols = FALSE,#
  labels_row = hit_labels,#
  labels_col = hit_labels,#
  main = "Chromosome 8 LD and shared GWAS hits"#
)
pheatmap::pheatmap(#
  ld_r2,#
  cluster_rows = FALSE,#
  cluster_cols = FALSE,#
  show_rownames = FALSE,#
  show_colnames = FALSE,#
  main = "Chromosome 8 local LD"#
)#
#
ld_r2[shared_snps, shared_snps, drop = FALSE]
pheatmap::pheatmap(#
  ld_r2,#
  cluster_rows = FALSE,#
  cluster_cols = FALSE,#
  show_rownames = FALSE,#
  show_colnames = FALSE,#
  main = "Chromosome 8 local LD"#
)
mlm_results[mlm_results$SNP %in% shared_snps,#
            c("SNP", "Chromosome", "Position", "P.value", "effect")]
shared_snps <- intersect(glm_hits$SNP, mlm_hits$SNP)#
#
ld_r2[shared_snps, shared_snps]
glm_common <- glm_results[#
  glm_results$SNP %in% shared_snps,#
  c("SNP", "Chromosome", "Position", "P.value", "effect")#
]#
#
mlm_common <- mlm_results[#
  mlm_results$SNP %in% shared_snps,#
  c("SNP", "P.value", "effect")#
]#
#
common_results <- merge(#
  glm_common,#
  mlm_common,#
  by = "SNP",#
  suffixes = c("_GLM", "_MLM")#
)#
#
common_results
names(glm_results) <- trimws(names(glm_results))#
names(mlm_results) <- trimws(names(mlm_results))
glm_common <- glm_results[#
  glm_results$SNP %in% shared_snps,#
  c("SNP", "Chromosome", "Position", "P.value", "effect"),#
  drop = FALSE#
]#
#
mlm_common <- mlm_results[#
  mlm_results$SNP %in% shared_snps,#
  c("SNP", "Chromosome", "Position", "P.value", "effect"),#
  drop = FALSE#
]#
#
common_results <- merge(#
  glm_common,#
  mlm_common,#
  by = "SNP",#
  suffixes = c("_GLM", "_MLM")#
)#
#
common_results
lead_snp <- mlm_results$SNP[#
  which.min(mlm_results$P.value)#
]#
#
lead_snp
nearby_results <- mlm_results[#
  mlm_results$Chromosome == 8 &#
    mlm_results$Position >= 60999071 &#
    mlm_results$Position <= 63279357,#
]#
#
nearby_results <- nearby_results[#
  order(nearby_results$P.value),#
]#
#
head(nearby_results, 20)
lead_genotype <- GD[, c("Taxa", lead_snp), drop = FALSE]#
#
names(lead_genotype)[2] <- "Lead_SNP"
lead_genotype <- lead_genotype[#
  match(Y_input$Taxa, lead_genotype$Taxa),#
  ,#
  drop = FALSE#
]#
#
stopifnot(identical(Y_input$Taxa, lead_genotype$Taxa))
conditional_covariate <- lead_genotype[, c("Taxa", "Lead_SNP")]#
#
conditional_result <- GAPIT(#
  Y = Y_input,#
  GD = GD,#
  GM = GM,#
  CV = conditional_covariate,#
  model = "MLM",#
  PCA.total = 3,#
  SNP.MAF = 0.01#
)
# Extract the numeric genotype matrix#
geno <- as.matrix(GD[, -1, drop = FALSE])#
storage.mode(geno) <- "numeric"#
#
sum(is.na(geno))#
sum(!is.finite(geno))
# Remove markers that contain no observed genotypes#
keep_markers <- colSums(is.finite(geno)) > 0#
#
geno <- geno[, keep_markers, drop = FALSE]#
GM <- GM[keep_markers, , drop = FALSE]#
#
# Mean-impute missing values marker by marker#
marker_means <- colMeans(geno, na.rm = TRUE)#
#
for (j in seq_len(ncol(geno))) {#
  missing <- !is.finite(geno[, j])#
#
  if (any(missing)) {#
    geno[missing, j] <- marker_means[j]#
  }#
}#
#
GD <- data.frame(#
  Taxa = GD$Taxa,#
  geno,#
  check.names = FALSE#
)
sum(is.na(GD[, -1]))#
sum(!is.finite(as.matrix(GD[, -1])))#
#
all(vapply(GD[, -1], is.numeric, logical(1)))
marker_names <- colnames(GD)[-1]#
#
GM <- GM[match(marker_names, GM$SNP), , drop = FALSE]#
#
stopifnot(#
  !anyNA(GM$SNP),#
  identical(marker_names, GM$SNP)#
)
conditional_result <- GAPIT(#
  Y = Y_input,#
  GD = GD,#
  GM = GM,#
  CV = conditional_covariate,#
  model = "MLM",#
  PCA.total = 3,#
  SNP.MAF = 0.01#
)
str(conditional_covariate)#
#
colSums(is.na(conditional_covariate))#
sapply(conditional_covariate[, -1, drop = FALSE], sd)
# Numeric genotype matrix: taxa x markers#
geno <- as.matrix(GD[, -1, drop = FALSE])#
storage.mode(geno) <- "numeric"#
#
# Remove markers with zero variance#
keep <- apply(geno, 2, sd, na.rm = TRUE) > 0#
geno <- geno[, keep, drop = FALSE]#
#
# Match genotype rows to phenotype order#
geno <- geno[match(Y_input$Taxa, GD$Taxa), , drop = FALSE]#
#
stopifnot(identical(rownames(geno), NULL) ||#
          nrow(geno) == nrow(Y_input))
pca <- prcomp(#
  geno,#
  center = TRUE,#
  scale. = TRUE#
)#
#
PC <- data.frame(#
  Taxa = Y_input$Taxa,#
  PC1 = pca$x[, 1],#
  PC2 = pca$x[, 2],#
  PC3 = pca$x[, 3]#
)
lead <- GD[, c("Taxa", lead_snp), drop = FALSE]#
names(lead)[2] <- "Lead_SNP"#
#
lead <- lead[match(Y_input$Taxa, lead$Taxa), , drop = FALSE]#
#
conditional_covariate <- merge(#
  PC,#
  lead,#
  by = "Taxa",#
  sort = FALSE#
)#
#
conditional_covariate <- conditional_covariate[#
  match(Y_input$Taxa, conditional_covariate$Taxa),#
  ,#
  drop = FALSE#
]#
#
stopifnot(identical(conditional_covariate$Taxa, Y_input$Taxa))
colSums(is.na(conditional_covariate))#
sapply(conditional_covariate[, -1, drop = FALSE], sd)
conditional_result <- GAPIT(#
  Y = Y_input,#
  GD = GD,#
  GM = GM,#
  CV = conditional_covariate,#
  model = "MLM",#
  PCA.total = 0,#
  SNP.MAF = 0.01#
)
conditional_results <- conditional_result$GWAS#
#
names(conditional_results)
alpha_bonferroni <- 0.05 / nrow(mlm_results)#
#
conditional_hits <- conditional_results[#
  conditional_results$P.value < alpha_bonferroni,#
  ,#
  drop = FALSE#
]#
#
conditional_hits
alpha_bonferroni <- 0.05 / nrow(mlm_results)#
#
conditional_hits <- conditional_results[#
  conditional_results$P.value < alpha_bonferroni,#
  ,#
  drop = FALSE#
]#
#
conditional_hits
conditional_candidates <- conditional_results[#
  conditional_results$SNP %in% shared_snps,#
  c("SNP", "Chromosome", "Position ", "P.value", "effect"),#
  drop = FALSE#
]#
#
conditional_candidates
comparison <- merge(#
  mlm_results[#
    mlm_results$SNP %in% shared_snps,#
    c("SNP", "P.value"),#
    drop = FALSE#
  ],#
  conditional_results[#
    conditional_results$SNP %in% shared_snps,#
    c("SNP", "P.value"),#
    drop = FALSE#
  ],#
  by = "SNP",#
  suffixes = c("_MLM", "_Conditional")#
)#
#
comparison$Significant_MLM <-#
  comparison$P.value_MLM < alpha_bonferroni#
#
comparison$Significant_Conditional <-#
  comparison$P.value_Conditional < alpha_bonferroni#
#
comparison
other_candidates <- setdiff(shared_snps, lead_snp)#
#
conditional_other <- conditional_results[#
  conditional_results$SNP %in% other_candidates,#
  ,#
  drop = FALSE#
]#
#
conditional_other[#
  order(conditional_other$P.value),#
  c("SNP", "P.value", "effect"),#
  drop = FALSE#
]
