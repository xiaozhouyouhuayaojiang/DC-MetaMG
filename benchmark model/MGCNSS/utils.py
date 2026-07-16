

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_recall_curve, roc_curve, auc
import torch_geometric.transforms as T
from torch_geometric.data import Data, HeteroData
import csv
import os


def set_seed(seed=42):

    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_features(feature_matrix, method='minmax'):

    if method == 'minmax':
        min_val = feature_matrix.min()
        max_val = feature_matrix.max()
        if max_val - min_val > 1e-10:
            return (feature_matrix - min_val) / (max_val - min_val)
        return feature_matrix
    elif method == 'zscore':
        mean = feature_matrix.mean()
        std = feature_matrix.std()
        if std > 1e-10:
            return (feature_matrix - mean) / std
        return feature_matrix
    else:
        return feature_matrix


def load_data(data_dir='D:\\Project_python\\CircRNA-MiRNA\\MGCNSS-master\\lncrna', type="None"):

    import os

    drug_features = np.loadtxt(os.path.join(data_dir, 'GIP_Topo_drug.txt'))
    ncrna_features = np.loadtxt(os.path.join(data_dir, f'Gaussian_sim_{type}.txt'))

    drug_features = normalize_features(drug_features, method='minmax')
    ncrna_features = normalize_features(ncrna_features, method='minmax')

    interactions_df = pd.read_csv(os.path.join(data_dir, 'edges.txt'), sep='\t', header=None)
    interactions = interactions_df.values  # (num_interactions, 2)

    return drug_features, ncrna_features, interactions


def create_sparse_adj(adj_matrix, threshold=0.0, normalize=True):

    adj_matrix = adj_matrix.copy()
    adj_matrix[adj_matrix < threshold] = 0

    if normalize:
        row_sum = adj_matrix.sum(axis=1)
        d_inv_sqrt = np.where(row_sum > 0, row_sum ** -0.5, 0.0)
        adj_matrix = d_inv_sqrt[:, None] * adj_matrix * d_inv_sqrt[None, :]

    adj_sparse = csr_matrix(adj_matrix)
    adj_coo = adj_sparse.tocoo()

    indices = torch.LongTensor(np.vstack([adj_coo.row, adj_coo.col]))
    values = torch.FloatTensor(adj_coo.data)

    return torch.sparse.FloatTensor(indices, values, torch.Size(adj_coo.shape))


def split_data(interactions, num_drugs, num_ncrnas, train_ratio=0.7, val_ratio=0.1, seed=42):

    torch.manual_seed(seed)

    num_total = num_drugs + num_ncrnas
    test_ratio = round(1.0 - train_ratio - val_ratio, 6)

    drug_idx = torch.tensor(interactions[:, 0], dtype=torch.long)
    ncrna_idx = torch.tensor(interactions[:, 1], dtype=torch.long) + num_drugs

    fwd = torch.stack([drug_idx, ncrna_idx], dim=0)
    bwd = torch.stack([ncrna_idx, drug_idx], dim=0)
    edge_index = torch.cat([fwd, bwd], dim=1)

    data = Data(x=torch.zeros(num_total, 1), edge_index=edge_index)

    transform = T.RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        split_labels=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
    )
    train_data, val_data, test_data = transform(data)

    def extract_bipartite(split_obj):
        def to_local(ei):
            src, dst = ei[0], ei[1]

            return np.column_stack([src, dst]).astype(int)

        return to_local(split_obj.pos_edge_label_index), to_local(split_obj.neg_edge_label_index)

    train_pos, train_neg = extract_bipartite(train_data)
    val_pos, val_neg = extract_bipartite(val_data)
    test_pos, test_neg = extract_bipartite(test_data)

    return train_pos, val_pos, test_pos, train_neg, val_neg, test_neg


def evaluate_model(model, predictor, embeddings, pos_edges, neg_edges, device):

    predictor.eval()

    with torch.no_grad():
        emb = embeddings.to(device)
        d = pos_edges[:, 0]
        n = pos_edges[:, 1]
        pos_src = emb[d]
        pos_dst = emb[n]
        pos_scores = torch.sigmoid(predictor(pos_src, pos_dst))

        neg_src = emb[neg_edges[:, 0]]
        neg_dst = emb[neg_edges[:, 1]]
        neg_scores = torch.sigmoid(predictor(neg_src, neg_dst))

        # Combine scores and labels
        all_scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
        all_labels = np.concatenate([
            np.ones(len(pos_edges)),
            np.zeros(len(neg_edges))
        ])

    # Compute metrics
    auc_score = roc_auc_score(all_labels, all_scores)

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    aupr = auc(recall, precision)

    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_scores)

    # Binary classification metrics (threshold = 0.5)
    pred_labels = (all_scores >= 0.5).astype(int)
    f1 = f1_score(all_labels, pred_labels)
    recall_score_val = recall_score(all_labels, pred_labels)
    acc = accuracy_score(all_labels, pred_labels)

    metrics = {
        'auc': auc_score,
        'f1': f1,
        'recall': recall_score_val,
        'accuracy': acc,
        'aupr': aupr,
        'fpr': fpr,
        'tpr': tpr
    }

    return metrics


def combine_features(drug_features, ncrna_features):

    # Pad to same dimension
    max_dim = max(drug_features.shape[1], ncrna_features.shape[1])

    drug_padded = np.pad(drug_features,
                         ((0, 0), (0, max_dim - drug_features.shape[1])),
                         mode='constant', constant_values=0)
    ncrna_padded = np.pad(ncrna_features,
                          ((0, 0), (0, max_dim - ncrna_features.shape[1])),
                          mode='constant', constant_values=0)

    # Concatenate: [drug_features; ncrna_features]
    combined = np.vstack([drug_padded, ncrna_padded])

    return torch.FloatTensor(combined)


def project_features(drug_features, ncrna_features, proj_dim=256):

    import torch.nn as nn
    drug_t = torch.FloatTensor(drug_features)
    ncrna_t = torch.FloatTensor(ncrna_features)

    drug_proj = nn.Linear(drug_features.shape[1], proj_dim, bias=False)
    ncrna_proj = nn.Linear(ncrna_features.shape[1], proj_dim, bias=False)
    nn.init.xavier_uniform_(drug_proj.weight)
    nn.init.xavier_uniform_(ncrna_proj.weight)

    with torch.no_grad():
        drug_out = drug_proj(drug_t)
        ncrna_out = ncrna_proj(ncrna_t)

    return torch.cat([drug_out, ncrna_out], dim=0)  # (N, proj_dim)


def build_bipartite_adj(train_pos, num_drugs, num_ncrnas):

    N = num_drugs + num_ncrnas
    drug_idx = train_pos[:, 0]                   # local drug indices
    ncrna_idx = train_pos[:, 1]     # global ncRNA indices

    # Symmetric bipartite: both directions
    rows = np.concatenate([drug_idx, ncrna_idx])
    cols = np.concatenate([ncrna_idx, drug_idx])
    vals = np.ones(len(rows), dtype=np.float32)

    # D^{-1/2} A D^{-1/2} normalization
    deg = np.zeros(N, dtype=np.float32)
    np.add.at(deg, rows, vals)
    d_inv_sqrt = np.where(deg > 0, deg ** -0.5, 0.0)
    norm_vals = d_inv_sqrt[rows] * vals * d_inv_sqrt[cols]

    indices = torch.LongTensor(np.vstack([rows, cols]))
    values = torch.FloatTensor(norm_vals)
    return torch.sparse.FloatTensor(indices, values, torch.Size([N, N]))


def print_metrics(split_name, metrics):
    """Pretty print evaluation metrics"""
    print(f"{split_name} Metrics:")
    print(f"  AUC:      {metrics['auc']:.4f}")
    print(f"  AUPR:     {metrics['aupr']:.4f}")
    print(f"  F1:       {metrics['f1']:.4f}")
    print(f"  Recall:   {metrics['recall']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")

