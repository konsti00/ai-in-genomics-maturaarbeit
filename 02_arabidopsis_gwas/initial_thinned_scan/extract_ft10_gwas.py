import csv
import gzip
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE.parent  # 02_arabidopsis_gwas: FT10_Phenotype.csv and 1001_SNP_MATRIX/
PHENOTYPE_FILE = DATA_DIR / 'FT10_Phenotype.csv'
HDF5_FILE = DATA_DIR / '1001_SNP_MATRIX' / 'imputed_snps_binary.hdf5'
OUTPUT_GENOTYPES = BASE / 'ft10_genotypes.csv.gz'
OUTPUT_MARKERS = BASE / 'ft10_marker_map.csv'
OUTPUT_PHENOTYPE = BASE / 'ft10_gapit_phenotype.csv'
MARKER_STEP = 20
BLOCK_SIZE = 2000


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

with h5py.File(HDF5_FILE, 'r') as hdf:
    accessions = [clean_id(value) for value in hdf['accessions'][:]]
    accession_to_index = {
        accession: index for index, accession in enumerate(accessions)
    }

    common_ids = [
        accession for accession in phenotype['accession_id']
        if accession in accession_to_index
    ]

    if len(common_ids) < 100:
        raise RuntimeError(
            f'Only {len(common_ids)} accessions matched. '
            'The accession IDs should be checked before GWAS.'
        )

    genotype_indices = np.array(
        [accession_to_index[accession] for accession in common_ids],
        dtype=np.int64,
    )

    positions = hdf['positions'][:]
    marker_indices = np.arange(0, len(positions), MARKER_STEP, dtype=np.int64)
    selected_positions = positions[marker_indices]

    chr_regions = np.asarray(hdf['positions'].attrs['chr_regions'])
    chromosome_ends = chr_regions[:, 1]
    selected_chromosomes = (
        np.searchsorted(chromosome_ends, marker_indices, side='right') + 1
    )

    marker_names = [
        f'Chr{chromosome}_{position}'
        for chromosome, position in zip(
            selected_chromosomes, selected_positions
        )
    ]

    with gzip.open(OUTPUT_GENOTYPES, 'wt', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['Marker'] + common_ids)

        for start in range(0, len(marker_indices), BLOCK_SIZE):
            block_indices = marker_indices[start:start + BLOCK_SIZE]
            block = hdf['snps'][block_indices, :]
            block = block[:, genotype_indices]

            for marker_name, row in zip(
                marker_names[start:start + BLOCK_SIZE], block
            ):
                writer.writerow([marker_name] + row.astype(np.int8).tolist())

with open(OUTPUT_MARKERS, 'w', newline='') as output:
    writer = csv.writer(output)
    writer.writerow(['Marker', 'Chr', 'Pos'])
    writer.writerows(
        zip(marker_names, selected_chromosomes, selected_positions)
    )

phenotype_lookup = phenotype.set_index('accession_id')
gapit_phenotype = pd.DataFrame({
    'Taxa': common_ids,
    'FT10': [phenotype_lookup.loc[accession, 'phenotype_value']
             for accession in common_ids],
})
gapit_phenotype.to_csv(OUTPUT_PHENOTYPE, index=False)

print(f'Phenotype records used: {len(phenotype)}')
print(f'Matched accessions: {len(common_ids)}')
print(f'Markers selected: {len(marker_indices)}')
print(f'Created: {OUTPUT_GENOTYPES}')
print(f'Created: {OUTPUT_MARKERS}')
print(f'Created: {OUTPUT_PHENOTYPE}')
