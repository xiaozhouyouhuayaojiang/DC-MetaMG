import numpy as np
from util.getValue import getValue
from util.util import set_seed, getMetaPath

set_seed(48)

if __name__ == "__main__":

    # ── Drug meta-path ────────────────────────────────────────────────────────
    drug_disease_matrix = np.loadtxt('data/array_DD.txt', dtype=int)
    num_D = drug_disease_matrix.shape[0]

    DDI = getMetaPath(
        num=num_D,
        object_disease_matrix=drug_disease_matrix,
        correlation_file='data/correlation_Drug.txt',
        max_hops=3,
        alpha=2.0,
        beta=1.0,
    )

    with open('../../data/LncRNA/DATA_DMP/MetaPath_drug.txt', 'w') as f:
        for row in DDI:
            f.write(' '.join(map(str, row)) + '\n')
    print(f"Drug meta-path saved: {DDI.sum()} edges")

    # ── miRNA meta-path ───────────────────────────────────────────────────────
    mirna_disease_matrix = np.loadtxt('data/array_LD.txt', dtype=int)
    num_mi = mirna_disease_matrix.shape[0]

    MMI = getMetaPath(
        num=num_mi,
        object_disease_matrix=mirna_disease_matrix,
        correlation_file='data/correlation_lncRNA.txt',
        max_hops=2,
        alpha=2.0,
        beta=1.0,
    )
    with open('../../data/LncRNA/DATA_DMP/MetaPath_lncRNA.txt', 'w') as f:
        for row in MMI:
            f.write(' '.join(map(str, row)) + '\n')
    print(f"lncRNA meta-path saved: {MMI.sum()} edges")
