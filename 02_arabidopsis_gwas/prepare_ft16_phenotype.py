"""Prepare the FT16 phenotype file used for the targeted FT16 analysis.

The FT16 values are cleaned and matched to the genotyped accessions in the
same way as FT10 in prepare_ft10_blocks.py. This step was originally run as
a one-off command; this script reproduces the original
ft16_gapit_phenotype.csv exactly (970 accessions, identical order and values).

Run from the 02_arabidopsis_gwas folder:
    python prepare_ft16_phenotype.py
"""

from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers the HDF5 compression filters)
import pandas as pd

BASE = Path(__file__).resolve().parent
PHENOTYPE_FILE = BASE / 'FT16_Phenotype.csv'
GENOTYPE_FILE = BASE / '1001_SNP_MATRIX' / 'imputed_snps_binary.hdf5'
OUTPUT_FILE = BASE / 'ft16_gapit_phenotype.csv'


def clean_id(value):
    if isinstance(value, bytes):
        value = value.decode()
    text = str(value).strip()
    if text.endswith('.0'):
        text = text[:-2]
    return text


phenotype = pd.read_csv(PHENOTYPE_FILE)
phenotype['accession_id'] = phenotype['accession_id'].map(clean_id)
phenotype['phenotype_value'] = pd.to_numeric(
    phenotype['phenotype_value'], errors='coerce'
)
phenotype = phenotype.dropna(subset=['accession_id', 'phenotype_value'])
phenotype = phenotype.drop_duplicates(subset=['accession_id'], keep='first')

with h5py.File(GENOTYPE_FILE, 'r') as genotype_hdf:
    genotyped = {clean_id(value) for value in genotype_hdf['accessions'][:]}

common_ids = [
    accession for accession in phenotype['accession_id']
    if accession in genotyped
]
phenotype_lookup = phenotype.set_index('accession_id')
gapit_phenotype = pd.DataFrame({
    'Taxa': common_ids,
    'FT16': [
        phenotype_lookup.loc[accession, 'phenotype_value']
        for accession in common_ids
    ],
})
gapit_phenotype.to_csv(OUTPUT_FILE, index=False)
print(f'Wrote {OUTPUT_FILE} with {len(gapit_phenotype)} accessions')
