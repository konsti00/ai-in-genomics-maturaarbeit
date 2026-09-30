# Tutorial: GWAS of bean disease resistance

This folder documents the introductory tutorial (thesis Section 2.1), in
which the GAPIT workflow was practised on the course exercise "GWAS Exercise
2: Bean Disease Resistance" (Molecular Plant Breeding lecture, FS25).

The tutorial was carried out interactively in the R console, so there is no
separate analysis script. [`tutorial_R_console_history.R`](tutorial_R_console_history.R)
is the unedited console history of these sessions. It shows the complete
work process, including package installation, data loading, quality control
(removal of markers and samples with more than 20% missing values),
matching of genotype and phenotype records, and the GAPIT runs with the
general linear model (GLM) and the mixed linear model (MLM). It also
contains exploratory and failed commands, and it is not intended to be run
from top to bottom.

The input files `extBALSIT_hmp.txt` and `ALS_phenotypes.txt` are course
material and are not included.
