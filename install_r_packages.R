# Installs the R packages used for the GAPIT analyses.
# The analyses were run with R 4.6.1 and GAPIT 4.1.0
# (commit 90127ea74f94cddca99ebb32c36f22e4a2e945dc).

install.packages(c("remotes", "BiocManager"))
BiocManager::install(c("multtest", "Biobase"))
remotes::install_github(
  "jiabowang/GAPIT",
  ref = "90127ea74f94cddca99ebb32c36f22e4a2e945dc"
)
