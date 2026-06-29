import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


file_path = 'MD_ARR.txt'

miRNA_disease_matrix = []
with open(file_path, 'r') as file:
    for line in file:
        row = line.strip().split()
        miRNA_disease_matrix.append(list(map(int, row)))

association_matrix = np.array(miRNA_disease_matrix)


file_path = 'DD_ARR.txt'

drug_disease_matrix = []
with open(file_path, 'r') as file:
    for line in file:
        row = line.strip().split()
        drug_disease_matrix.append(list(map(int, row)))

association_matrix = np.array(drug_disease_matrix)



miRNA_disease_sparse = csr_matrix(miRNA_disease_matrix)
drug_disease_sparse = csr_matrix(drug_disease_matrix)

miRNA_disease_associations = miRNA_disease_sparse.nonzero()
drug_disease_associations = drug_disease_sparse.nonzero()

miRNA_disease_map = {}
for miRNA_idx, disease_idx in zip(*miRNA_disease_associations):
    if miRNA_idx not in miRNA_disease_map:
        miRNA_disease_map[miRNA_idx] = []
    miRNA_disease_map[miRNA_idx].append(disease_idx)

drug_disease_map = {}
for drug_idx, disease_idx in zip(*drug_disease_associations):
    if drug_idx not in drug_disease_map:
        drug_disease_map[drug_idx] = []
    drug_disease_map[drug_idx].append(disease_idx)


def extract_paths(miRNA_disease_map, drug_disease_map):
    miRNA_miRNA_paths = []
    drug_drug_paths = []
    miRNA_drug_paths = []
    drug_miRNA_paths = []

    # miRNA → disease → miRNA
    for miRNA1 in miRNA_disease_map:
        for disease in miRNA_disease_map[miRNA1]:
            for miRNA2 in miRNA_disease_map:
                if miRNA1 != miRNA2 and disease in miRNA_disease_map[miRNA2]:
                    miRNA_miRNA_paths.append((miRNA1, miRNA2))

    # drug → disease → drug
    for drug1 in drug_disease_map:
        for disease in drug_disease_map[drug1]:
            for drug2 in drug_disease_map:
                if drug1 != drug2 and disease in drug_disease_map[drug2]:
                    drug_drug_paths.append((drug1, drug2))

    # miRNA → disease → drug
    for miRNA in miRNA_disease_map:
        for disease in miRNA_disease_map[miRNA]:
            for drug in drug_disease_map:
                if disease in drug_disease_map[drug]:
                    miRNA_drug_paths.append((miRNA, drug))

    # drug → disease → miRNA
    for drug in drug_disease_map:
        for disease in drug_disease_map[drug]:
            for miRNA in miRNA_disease_map:
                if disease in miRNA_disease_map[miRNA]:
                    drug_miRNA_paths.append((drug, miRNA))

    return miRNA_miRNA_paths, drug_drug_paths, miRNA_drug_paths, drug_miRNA_paths


miRNA_miRNA_paths, drug_drug_paths, miRNA_drug_paths, drug_miRNA_paths = extract_paths(miRNA_disease_map, drug_disease_map)


print("miRNA-miRNA paths:", miRNA_miRNA_paths)
print("drug-drug paths:", drug_drug_paths)
print("miRNA-drug paths:", miRNA_drug_paths)
print("drug-miRNA paths:", drug_miRNA_paths)
