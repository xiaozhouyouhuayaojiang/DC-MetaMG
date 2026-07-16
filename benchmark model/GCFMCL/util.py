import numpy as np
import torch
import random
from torch_geometric.data import Data
import torch_geometric.transforms as T
from collections import deque
import pandas as pd
from torch_geometric.data import HeteroData
import os

def data_preparation(data_path):
    """
    Load LncRNA-Drug data with features and edges

    Args:
        data_path: Path to data directory

    Returns:
        train_data: Training data split
        test_data: Test data split
        num_lncrnas: Number of LncRNA nodes
        num_drugs: Number of Drug nodes
    """
    # Load LncRNA features (396 dimensions)
    lncrna_features = []
    with open(f'dataset/{data_path}/Feature_{data_path}_Sq.txt', 'r') as f:
        for line in f:
            features = [float(x) for x in line.strip().split()]
            lncrna_features.append(features)
    lncrna_features = np.array(lncrna_features)

    # Load Drug features (396 dimensions)
    drug_features = []
    with open(f'dataset/{data_path}/drug_feature.txt', 'r') as f:
        for line in f:
            features = [float(x) for x in line.strip().split()]
            drug_features.append(features)
    drug_features = np.array(drug_features)

    num_lncrnas = lncrna_features.shape[0]
    num_drugs = drug_features.shape[0]

    # Concatenate features: [lncrna_features; drug_features]
    all_features = np.vstack([drug_features, lncrna_features ])
    x = torch.FloatTensor(all_features)

    # Load edges (lncrna_idx, drug_idx)
    edges = []
    with open(f'dataset/{data_path}/edges.txt', 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                lncrna_idx = int(parts[1])
                drug_idx = int(parts[0])
                # Offset drug index by num_lncrnas to create global indexing
                edges.append([drug_idx, lncrna_idx + num_drugs])

    edges = np.array(edges).T
    edge_index = torch.LongTensor(edges)

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index)

    # Apply RandomLinkSplit as specified
    transform = T.RandomLinkSplit(
        num_val=0,
        num_test=0.2,
        is_undirected=True,
        split_labels=True,
        add_negative_train_samples=True
    )


    train_data, _, test_data = transform(data)
    print(f"Data loaded: {num_lncrnas} LncRNAs, {num_drugs} Drugs")
    print(f"Total nodes: {x.shape[0]}, Feature dim: {x.shape[1]}")
    print(f"Train edges: {train_data.pos_edge_label_index.shape[1]}")
    print(f"Test edges: {test_data.pos_edge_label_index.shape[1]}")



    return train_data, test_data