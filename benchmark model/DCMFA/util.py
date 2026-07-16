import numpy as np
import random
import torch
import pandas as pd
from model import laplacian_pe

np.random.seed(2)
random.seed(2)


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def load_data(data_type: str):
    """
    Load embeddings and edge list for a given data type (e.g. 'mirna', 'lncRNA').

    Expects under  data/<data_type>/:
        embedding1.txt  – miRNA features  (N_mirna × dim1)
        embedding2.txt  – drug   features  (N_drug  × dim2)
        edges.txt       – positive pairs   "mirna_idx drug_idx"

    Returns
    -------
    pos_data : list of [emb_mirna, emb_drug]
    neg_data : list of [emb_mirna, emb_drug]
    """
    embedding1 = np.loadtxt(f'data/{data_type}/embedding1.txt', dtype=float, delimiter=' ')
    embedding2 = np.loadtxt(f'data/{data_type}/embedding2.txt', dtype=float, delimiter=' ')

    edges = []
    with open(f'data/{data_type}/edges.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v = map(int, line.split())
            edges.append([u, v])

    edge_set = set(map(tuple, edges))

    pos_data, neg_data = [], []
    for i in range(len(embedding1)):
        for j in range(len(embedding2)):
            if (i, j) in edge_set:
                pos_data.append([embedding1[i], embedding2[j]])
            else:
                neg_data.append([embedding1[i], embedding2[j]])

    return pos_data, neg_data


def load_dual_channel_data(type_ch1: str):
    """
    Load data for the dual-channel DCMFA model.

    Channel 1 (FAT): miRNA k-mer sequence ↔ drug bond-angle
    Channel 2 (FAV): miRNA func-similarity ↔ drug biological features

    Both channels must share the same positive/negative pair indices.

    Parameters
    ----------
    type_ch1 : folder name for channel-1 data  (e.g. 'mirna')
    type_ch2 : folder name for channel-2 data  (e.g. 'lncRNA')

    Returns
    -------
    pos_ch1, neg_ch1 : list of [emb1, emb2]  – channel-1 pairs
    pos_ch2, neg_ch2 : list of [emb1, emb2]  – channel-2 pairs
      (indices are aligned: pos_ch1[i] and pos_ch2[i] belong to the same pair)
    adj_mirna : (N_mirna, N_mirna) adjacency for Laplacian PE (from edges)
    adj_drug  : (N_drug,  N_drug)  adjacency for Laplacian PE
    """
    emb1_ch1 = np.loadtxt(f'data/{type_ch1}/embedding1.txt', dtype=float, delimiter=' ')
    emb2_ch1 = np.loadtxt(f'data/{type_ch1}/embedding2.txt', dtype=float, delimiter=' ')
    emb1_ch2 = np.loadtxt(f'data/{type_ch1}/embedding3.txt', dtype=float, delimiter=' ')
    emb2_ch2 = np.loadtxt(f'data/{type_ch1}/embedding4.txt', dtype=float, delimiter=' ')

    # Use channel-1 edges as the authoritative pair list
    edges = []
    with open(f'data/{type_ch1}/edges.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v = map(int, line.split())
            edges.append([u, v])

    edge_set = set(map(tuple, edges))

    # Build adjacency matrices for Laplacian PE
    N_drug = len(emb1_ch1)
    N_mirna  = len(emb2_ch1)
    adj_mirna = np.zeros((N_mirna, N_mirna))
    adj_drug  = np.zeros((N_drug,  N_drug))
    for u, v in edges:
        # miRNA-miRNA co-occurrence proxy: self-loop weight
        adj_mirna[v, v] += 1
        adj_drug[u, u]  += 1

    pos_ch1, neg_ch1 = [], []
    pos_ch2, neg_ch2 = [], []

    for i in range(N_drug):
        for j in range(N_mirna):
            if (i, j) in edge_set:
                pos_ch1.append([emb1_ch1[i], emb2_ch1[j]])
                pos_ch1.append([emb2_ch1[j], emb1_ch1[i]])

                pos_ch2.append([emb1_ch2[i], emb2_ch2[j]])
                pos_ch2.append([emb2_ch2[j], emb1_ch2[i]])
            else:
                neg_ch1.append([emb1_ch1[i], emb2_ch1[j]])
                neg_ch1.append([ emb2_ch1[j], emb1_ch1[i]])

                neg_ch2.append([emb1_ch2[i], emb2_ch2[j]])
                neg_ch2.append([emb2_ch2[j], emb1_ch2[i]])

    return pos_ch1, neg_ch1, pos_ch2, neg_ch2, adj_mirna, adj_drug


# ─────────────────────────────────────────────────────────────
# Balancing
# ─────────────────────────────────────────────────────────────

def split_data(pos_data, neg_data):
    """Down-sample negatives to match the number of positives."""
    n_pos = len(pos_data)
    neg_list = []
    times = n_pos
    while times:
        index = np.random.randint(len(neg_data))
        neg_list.append(neg_data[index])
        del neg_data[index]
        times -= 1
    return pos_data, neg_list


def split_data_dual(pos_ch1, neg_ch1, pos_ch2, neg_ch2):
    """Balance both channels simultaneously (same indices)."""
    n_pos = len(pos_ch1)
    indices = list(range(len(neg_ch1)))
    chosen = random.sample(indices, n_pos)
    chosen_set = sorted(chosen)
    neg_ch1_bal = [neg_ch1[i] for i in chosen_set]
    neg_ch2_bal = [neg_ch2[i] for i in chosen_set]
    return pos_ch1, neg_ch1_bal, pos_ch2, neg_ch2_bal


# ─────────────────────────────────────────────────────────────
# Embedding utilities
# ─────────────────────────────────────────────────────────────

def deal_embedding(lst):
    """Concatenate [emb_a, emb_b] pairs into flat vectors."""
    return [np.concatenate([item[0], item[1]], axis=0) for item in lst]


# ─────────────────────────────────────────────────────────────
# Train/test split (5-fold)
# ─────────────────────────────────────────────────────────────

def split_train_test(pos_list, neg_list, fold_idx: int):
    """Standard 5-fold split used by original code."""
    train_data, train_label, test_data, test_label = [], [], [], []
    for index in range(len(pos_list)):
        if index % 5 == fold_idx:
            test_data.append(pos_list[index])
            test_label.append(1)
            test_data.append(neg_list[index])
            test_label.append(0)
        else:
            train_data.append(pos_list[index])
            train_label.append(1)
            train_data.append(neg_list[index])
            train_label.append(0)

    train_data  = torch.tensor(train_data,  dtype=torch.float32)
    train_label = torch.tensor(train_label, dtype=torch.float32)
    test_data   = torch.tensor(test_data,   dtype=torch.float32)
    test_label  = torch.tensor(test_label,  dtype=torch.float32)
    return train_data, train_label, test_data, test_label


def split_train_test_dual(pos_ch1, neg_ch1, pos_ch2, neg_ch2,
                          adj_drug: np.ndarray, adj_mirna: np.ndarray,
                          fold_idx: int, pe_k: int = 16):
    """
    5-fold split for dual-channel data.

    Returns tensors for both channels plus per-sample Laplacian PE.
    """


    def _split(pos, neg):
        tr, tr_l, te, te_l = [], [], [], []
        for idx in range(len(pos)):
            if idx % 5 == fold_idx:
                te.append(pos[idx]);  te_l.append(1)
                te.append(neg[idx]);  te_l.append(0)
            else:
                tr.append(pos[idx]);  tr_l.append(1)
                tr.append(neg[idx]);  tr_l.append(0)
        return tr, tr_l, te, te_l

    tr1, tr_l, te1, te_l = _split(pos_ch1, neg_ch1)
    tr2, _,    te2, _    = _split(pos_ch2, neg_ch2)

    def to_tensor(lst):
        a = torch.tensor([item[0] for item in lst], dtype=torch.float32)
        b = torch.tensor([item[1] for item in lst], dtype=torch.float32)
        return a, b

    tr_d1, tr_m1 = to_tensor(tr1)
    te_d1 ,te_m1= to_tensor(te1)
    tr_d2, tr_m2 = to_tensor(tr2)
    te_d2, te_m2 = to_tensor(te2)

    tr_label = torch.tensor(tr_l, dtype=torch.float32)
    te_label = torch.tensor(te_l, dtype=torch.float32)

    # Build per-sample PE: replicate node PE for each sample
    # Each sample is (mirna_i, drug_j); we keep one global mean PE per sample
    # (since we don't track indices here, use mean across all nodes as fallback)


    def expand_pe(mean_pe, n):
        return mean_pe.expand(n, -1)


    return (tr_d1, tr_m1,tr_d2, tr_m2, tr_label,
            te_d1, te_m1,  te_d2, te_m2, te_label)


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

def calculate_metrics(y_true, y_pred):
    TP = TN = FP = FN = 0
    for i in range(len(y_true)):
        yt, yp = int(y_true[i]), int(y_pred[i])
        if yt == 1 and yp == 1: TP += 1
        if yt == 0 and yp == 0: TN += 1
        if yt == 0 and yp == 1: FP += 1
        if yt == 1 and yp == 0: FN += 1
    accuracy    = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision   = TP / (TP + FP) if (TP + FP) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    return accuracy, sensitivity, precision, specificity
