results <- read.csv("ft10_FT10_GWAS_all.csv", stringsAsFactors = FALSE)
results <- results[order(results$Chromosome, results$Position), c("Chromosome", "Position", "P.value")]
results$minus_log10_P <- -log10(results$P.value)

chromosomes <- sort(unique(results$Chromosome))
offsets <- numeric(length(chromosomes))
if (length(chromosomes) > 1) {
  for (index in 2:length(chromosomes)) {
    previous <- chromosomes[index - 1]
    offsets[index] <- offsets[index - 1] + max(results$Position[results$Chromosome == previous])
  }
}
results$plot_x <- results$Position + offsets[match(results$Chromosome, chromosomes)]
centers <- sapply(seq_along(chromosomes), function(index) {
  values <- results$plot_x[results$Chromosome == chromosomes[index]]
  mean(range(values))
})

png("ft10_FT10_Manhattan.png", width = 2200, height = 900, res = 160)
plot(results$plot_x, results$minus_log10_P,
     pch = 16, cex = 0.25,
     col = rep(c("#174a5b", "#d66b3d"), length.out = nrow(results)),
     xaxt = "n", xlab = "Chromosome", ylab = expression(-log[10](P)),
     main = "FT10 GAPIT MLM genome-wide association")
axis(1, at = centers, labels = chromosomes)
abline(h = -log10(0.05 / nrow(results)), col = "#8b2f2f", lty = 2)
dev.off()

expected <- -log10((seq_len(nrow(results)) - 0.5) / nrow(results))
observed <- sort(results$minus_log10_P, decreasing = TRUE)
limit <- max(c(expected, observed))
png("ft10_FT10_QQ.png", width = 1000, height = 1000, res = 160)
plot(expected, observed, pch = 16, cex = 0.35, col = "#174a5b",
     xlab = "Expected -log10(P)", ylab = "Observed -log10(P)",
     main = "FT10 QQ plot", xlim = c(0, limit), ylim = c(0, limit))
abline(0, 1, col = "#8b2f2f", lty = 2)
dev.off()

cat("Wrote ft10_FT10_Manhattan.png and ft10_FT10_QQ.png\n")
