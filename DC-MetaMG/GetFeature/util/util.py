import os
import numpy as np
import torch
import torch_geometric.transforms as T
import pandas as pd
import random
from torch_geometric.data import Data


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_data(args):
    """
    Load drug-ncRNA interaction data and perform train/test split using RandomLinkSplit.
    The split is persisted to disk so that DC_Prediction can use the exact same partition.

    Returns:
        smiles_list: list of drug SMILES strings
        splits: dict with keys 'train' and 'test' (PyG Data objects)
    """
    print("Loading data.")

    drug_ncRNA = pd.read_csv(os.path.join(args.data_path, 'edges.txt'), header=None, sep='\t')
    drug_list = list(range(args.n_Drug))
    mirna_list = list(range(args.n_NcRNA))

    # Build edge index: drug nodes [0, n_Drug), ncRNA nodes [n_Drug, n_Drug+n_NcRNA)
    adj = torch.LongTensor(
        [[drug_list.index(int(x[0])), mirna_list.index(int(x[1])) + len(drug_list)]
         for x in drug_ncRNA.values]
    ).T

    feature = torch.zeros((args.n_NcRNA + args.n_Drug, 1))
    node_types = torch.cat([
        torch.zeros(len(drug_list), dtype=torch.long),
        torch.ones(len(mirna_list), dtype=torch.long)
    ])
    node_index = torch.cat([
        torch.arange(len(drug_list)),
        torch.arange(len(mirna_list)) + len(drug_list)
    ])

    split_path = os.path.join(args.data_path, 'splits.pt')
    if False and os.path.exists(split_path):
        print(f"Loading existing splits from {split_path}")
        splits = torch.load(split_path)
    else:
        graph = Data(
            x=feature,
            node_index=node_index,
            edge_index=adj,
            type=node_types,
            edge_attr=None
        ).cuda()

        train_data, _, test_data = T.RandomLinkSplit(
            num_val=0,
            num_test=0.2,
            is_undirected=True,
            split_labels=True,
            add_negative_train_samples=True,
            neg_sampling_ratio=1.0
        )(graph)

        splits = dict(train=train_data, test=test_data)
        torch.save(splits, split_path)
        print(f"Saved splits to {split_path}")

    smiles_list = []
    with open(os.path.join(args.data_path, 'DrugSmile.csv'), 'r') as f:
        for line in f.readlines():
            smiles_list.append(line.strip())

    return smiles_list, splits


def get_edge_index(args, dataset, mode='MG'):
    """
    Build edge index for graph convolution.

    Args:
        args: argument namespace with data_path, n_Drug, n_NcRNA
        mode: 'MetaPath' uses meta-path adjacency; otherwise uses raw interaction edges

    Returns:
        edge_index: torch.LongTensor of shape [2, E]
    """
    n_nodes = args.n_Drug + args.n_NcRNA
    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=int)

    for i in range(n_nodes):
        adj_matrix[i, i] = 1

    '''interactions = pd.read_csv(os.path.join(args.data_path, "edges.txt"), header=None, sep='\t')
    for d, r in interactions.itertuples(index=False):
        adj_matrix[int(d), int(r) + args.n_Drug] = 1
        adj_matrix[int(r) + args.n_Drug, int(d)] = 1'''
    interactions = dataset.edge_index
    for d, r in interactions.T:
        adj_matrix[int(d), int(r)] = 1
        adj_matrix[int(r), int(d)] = 1


    if mode == 'MetaPath':
        meta_drug_path = os.path.join(args.data_path, "DATA_DMP/MetaPath_drug.txt")
        meta_ncrna_path = os.path.join(args.data_path, f"DATA_DMP/MetaPath_{args.dataset}.txt")

        meta_drug = []
        with open(meta_drug_path, 'r') as f:
            for line in f:
                meta_drug.append(list(map(int, line.strip().split())))
        meta_drug = np.array(meta_drug)

        meta_ncrna = []
        with open(meta_ncrna_path, 'r') as f:
            for line in f:
                meta_ncrna.append(list(map(int, line.strip().split())))
        meta_ncrna = np.array(meta_ncrna)

        adj_matrix[:meta_drug.shape[0], :meta_drug.shape[1]] = meta_drug
        adj_matrix[args.n_Drug:args.n_Drug + meta_ncrna.shape[0],
                   args.n_Drug:args.n_Drug + meta_ncrna.shape[1]] = meta_ncrna

    edge_index = torch.tensor(np.where(adj_matrix > 0), dtype=torch.long)
    return edge_index
