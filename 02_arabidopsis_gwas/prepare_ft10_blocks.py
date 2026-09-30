import csv
import gzip
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
PHENOTYPE_FILE = BASE / 'FT10_Phenotype.csv'
GENOTYPE_FILE = BASE / '1001_SNP_MATRIX' / 'imputed_snps_binary.hdf5'
KINSHIP_FILE = BASE / '1001_SNP_MATRIX' / 'kinship_ibs_mac5.hdf5'
BLOCK_DIR = BASE / 'ft10_blocks'
BLOCK_SIZE = 100_000
MIN_MAF = 0.01


def clean_id(value):
    if isinstance(value, bytes):
        value = value.decode()
    text = str(value).strip()
    if text.endswith('.0'):
        text = text[:-2]
    return text


BLOCK_DIR.mkdir(exist_ok=True)
phenotype = pd.read_csv(PHENOTYPE_FILE)
phenotype['accession_id'] = phenotype['accession_id'].map(clean_id)
phenotype['phenotype_value'] = pd.to_numeric(
    phenotype['phenotype_value'], errors='coerce'
)
phenotype = phenotype.dropna(subset=['accession_id', 'phenotype_value'])
phenotype = phenotype.drop_duplicates(subset=['accession_id'], keep='first')

with h5py.File(GENOTYPE_FILE, 'r') as genotype_hdf, h5py.File(
    KINSHIP_FILE, 'r'
) as kinship_hdf:
    genotype_accessions = [
        clean_id(value) for value in genotype_hdf['accessions'][:]
    ]
    accession_to_index = {
        accession: index
        for index, accession in enumerate(genotype_accessions)
    }

    common_ids = [
        accession for accession in phenotype['accession_id']
        if accession in accession_to_index
    ]
    if len(common_ids) < 100:
        raise RuntimeError(f'Only {len(common_ids)} accessions matched.')

    genotype_indices = np.array(
        [accession_to_index[accession] for accession in common_ids],
        dtype=np.int64,
    )

    phenotype_lookup = phenotype.set_index('accession_id')
    gapit_phenotype = pd.DataFrame({
        'Taxa': common_ids,
        'FT10': [
            phenotype_lookup.loc[accession, 'phenotype_value']
            for accession in common_ids
        ],
    })
    gapit_phenotype.to_csv(BASE / 'ft10_gapit_phenotype.csv', index=False)

    kinship_accessions = [
        clean_id(value) for value in kinship_hdf['accessions'][:]
    ]
    kinship_index = {
        accession: index
        for index, accession in enumerate(kinship_accessions)
    }
    kinship_indices = [kinship_index[accession] for accession in common_ids]
    kinship_full = kinship_hdf['kinship'][:]
    kinship = kinship_full[np.ix_(kinship_indices, kinship_indices)]
    kinship_df = pd.DataFrame(kinship, index=common_ids, columns=common_ids)
    kinship_df.to_csv(BASE / 'ft10_kinship.csv')

    positions = genotype_hdf['positions'][:]
    chr_regions = np.asarray(genotype_hdf['positions'].attrs['chr_regions'])
    chromosome_ends = chr_regions[:, 1]
    chromosomes = (
        np.searchsorted(chromosome_ends, np.arange(len(positions)), side='right')
        + 1
    )

    for chromosome in range(1, 6):
        chromosome_indices = np.flatnonzero(chromosomes == chromosome)
        for block_number, start in enumerate(
            range(0, len(chromosome_indices), BLOCK_SIZE), start=1
        ):
            marker_indices = chromosome_indices[start:start + BLOCK_SIZE]
            block = genotype_hdf['snps'][marker_indices, :]
            block = block[:, genotype_indices].astype(np.int8)

            allele_frequency = block.mean(axis=1)
            maf = np.minimum(allele_frequency, 1 - allele_frequency)
            keep = (maf >= MIN_MAF) & (maf <= 0.5)
            block = block[keep]
            kept_indices = marker_indices[keep]

            if block.shape[0] == 0:
                continue

            block_name = f'ft10_chr{chromosome}_block{block_number:03d}'
            genotype_path = BLOCK_DIR / f'{block_name}.csv.gz'
            marker_path = BLOCK_DIR / f'{block_name}_markers.csv'

            marker_names = [
                f'Chr{chromosome}_{positions[index]}'
                for index in kept_indices
            ]

            with gzip.open(genotype_path, 'wt', newline='') as output:
                writer = csv.writer(output)
                writer.writerow(['Marker'] + common_ids)
                for marker_name, row in zip(marker_names, block):
                    writer.writerow([marker_name] + row.tolist())

            with open(marker_path, 'w', newline='') as output:
                writer = csv.writer(output)
                writer.writerow(['Marker', 'Chr', 'Pos'])
                for marker_name, index in zip(marker_names, kept_indices):
                    writer.writerow([marker_name, chromosome, positions[index]])

            print(
                f'Created {block_name}: '
                f'{block.shape[0]} markers from {len(marker_indices)}'
            )

print(f'Matched accessions: {len(common_ids)}')
print(f'Block directory: {BLOCK_DIR}')
print('Created ft10_gapit_phenotype.csv and ft10_kinship.csv')
