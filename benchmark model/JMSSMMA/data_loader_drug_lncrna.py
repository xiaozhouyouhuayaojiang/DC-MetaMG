import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

def load_drug_lncrna_data(data_dir):

    # Sequence-based features only (no dependency on the association matrix)
    drug_features  = np.loadtxt(f'{data_dir}/drug_feature.txt')
    ncrna_features = np.loadtxt(f'{data_dir}/Feature_{data_dir}_Sq.txt')

    all_features = np.vstack([drug_features, ncrna_features])
    all_features = torch.tensor(all_features, dtype=torch.float)
    print(f"Loaded drug features: {drug_features.shape}")
    print(f"Loaded ncRNA features: {ncrna_features.shape}")

    # Load edges (format: drug_idx  ncRNA_idx)
    drug_ncRNA = pd.read_csv(f'{data_dir}/edges.txt', sep='\t', header=None)
    n_drug  = len(drug_features)
    n_ncrna = len(ncrna_features)
    drug_list  = list(range(n_drug))
    mirna_list = list(range(n_ncrna))

    # Build edge index: drug nodes [0, n_drug), ncRNA nodes [n_drug, n_drug+n_ncrna)
    adj = torch.LongTensor(
        [[drug_list.index(int(x[0])), mirna_list.index(int(x[1])) + n_drug]
         for x in drug_ncRNA.values]
    ).T

    data = Data(x=all_features, edge_index=adj, n_drug=n_drug, n_ncrna=n_ncrna)

    return data
