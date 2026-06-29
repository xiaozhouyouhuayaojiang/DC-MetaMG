import os
import numpy as np
import random
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_embeddings(dataset: str = 'MiRNA'):
    """
    Load the four embeddings produced by GetFeature/trainer.py.

    Args:
        dataset: 'MiRNA' or 'LncRNA'

    Returns:
        Tuple of four numpy arrays: (Drug_GT, NcRNA_GS, Drug_MG, NcRNA_SQ)
    """
    base = os.path.join('data', dataset)
    embedding1 = np.loadtxt(os.path.join(base, 'Drug_GT.txt'), dtype=float, delimiter=" ")
    embedding2 = np.loadtxt(os.path.join(base, 'NcRNA_GS.txt'), dtype=float, delimiter=" ")
    embedding3 = np.loadtxt(os.path.join(base, 'Drug_MG.txt'), dtype=float, delimiter=" ")
    embedding4 = np.loadtxt(os.path.join(base, 'NcRNA_SQ.txt'), dtype=float, delimiter=" ")

    emb_coa = np.vstack((embedding1, embedding2))
    emb_fin = np.vstack((embedding3, embedding4))

    #emb_coa = np.random.randn(len(embedding1) + len(embedding2), len(embedding1[0]))
    #emb_fin = np.random.randn(len(embedding1) + len(embedding2), len(embedding1[0]))
    return emb_coa, emb_fin


def load_edge_splits(dataset: str = 'MiRNA'):
    """
    Load the edge splits saved by GetFeature/trainer.py so that DC_Prediction
    uses the same train/test partition as the feature extraction stage.

    Args:
        dataset: 'MiRNA' or 'LncRNA'

    Returns:
        dict with keys:
            train_pos_edge_index, train_neg_edge_index,
            test_pos_edge_index,  test_neg_edge_index
    """
    split_path = os.path.join('data', dataset, 'edge_splits.pt')
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Edge splits file not found: {split_path}\n"
            "Please run GetFeature/main.py first to generate the splits."
        )

    return torch.load(split_path)


def build_pairs(emb_coa, emb_fin,
                edge_index: torch.Tensor, n_drug: int):
    """
    Build feature pairs from an edge index.

    Each edge (d, r) in edge_index maps drug d and ncRNA (r - n_drug) to
    their concatenated embedding vectors (two parallel feature channels).

    Args:
        embedding1, embedding2: coarse-grained drug / ncRNA embeddings
        embedding3, embedding4: fine-grained drug / ncRNA embeddings
        edge_index: LongTensor [2, E] with drug in row 0, ncRNA in row 1
        n_drug: number of drug nodes (offset for ncRNA indices)

    Returns:
        pairs, pairs1: lists of [drug_emb, ncrna_emb] pairs for both channels
    """
    pairs, pairs1 = [], []
    for i in range(edge_index.size(1)):
        a = edge_index[0, i].item()
        b = edge_index[1, i].item()

        '''
        if (a < 154 and b < 154) or (a > 154 and b > 154) :
            emb_dim = len(emb_coa[0])
            zero_vector = torch.zeros(emb_dim)
            pairs.append([zero_vector, zero_vector])
            pairs1.append([zero_vector, zero_vector])
        else:'''
        pairs.append([emb_coa[a], emb_coa[b]])

        pairs1.append([emb_fin[a], emb_fin[b]])

    return pairs, pairs1


def deal_embedding(pair_list, pair_list1):
    """
    Concatenate drug and ncRNA embeddings within each pair.

    Returns:
        Two lists of concatenated feature vectors.
    """
    result = [np.concatenate([p[0], p[1]], axis=0) for p in pair_list]
    result1 = [np.concatenate([p[0], p[1]], axis=0) for p in pair_list1]
    return result, result1


def kfold_split(pos_data, neg_data, pos_data1, neg_data1, fold: int, n_folds: int = 5):
    """
    Split training-set pairs into train/validation subsets for one fold.

    Positive and negative samples are shuffled with a fixed permutation so that
    results are reproducible across folds.

    Args:
        pos_data, neg_data: list of concatenated feature vectors (channel 1)
        pos_data1, neg_data1: same for channel 2
        fold: current fold index [0, n_folds)
        n_folds: total number of folds

    Returns:
        train_data, train_label, val_data, val_label — all as CUDA tensors
    """
    # Deterministic shuffle
    idx_pos = np.random.permutation(len(pos_data))
    idx_neg = np.random.permutation(len(neg_data))

    pos_data  = [pos_data[i]  for i in idx_pos]
    pos_data1 = [pos_data1[i] for i in idx_pos]
    neg_data  = [neg_data[i]  for i in idx_neg]
    neg_data1 = [neg_data1[i] for i in idx_neg]

    tr0, tr1, tr_lbl = [], [], []
    va0, va1, va_lbl = [], [], []

    for idx in range(len(pos_data)):
        if idx % n_folds == fold:
            va0.extend([pos_data[idx], neg_data[idx]])
            va1.extend([pos_data1[idx], neg_data1[idx]])
            va_lbl.extend([1, 0])
        else:
            tr0.extend([pos_data[idx], neg_data[idx]])
            tr1.extend([pos_data1[idx], neg_data1[idx]])
            tr_lbl.extend([1, 0])

    train_data  = torch.tensor(np.stack([tr0, tr1]),  dtype=torch.float32)
    val_data    = torch.tensor(np.stack([va0, va1]),  dtype=torch.float32)
    train_label = torch.tensor(tr_lbl, dtype=torch.float32)
    val_label   = torch.tensor(va_lbl, dtype=torch.float32)

    # Shuffle training samples
    perm = np.random.permutation(train_data.size(1))
    train_data[0] = train_data[0][perm]
    train_data[1] = train_data[1][perm]
    train_label   = train_label[perm]

    device = torch.device('cuda:0')
    return (train_data.to(device), train_label.to(device),
            val_data.to(device),   val_label.to(device))


def calculate_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    Compute accuracy, sensitivity (recall), precision, and specificity.

    Args:
        y_true: ground-truth binary labels
        y_pred: predicted binary labels (thresholded)

    Returns:
        accuracy, sensitivity, precision, specificity
    """
    y_true = y_true.cpu().numpy().astype(int)
    y_pred = y_pred.cpu().numpy().astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return accuracy, sensitivity, precision, specificity


class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(self, patience: int = 100, verbose: bool = False, delta: float = 0.0):
        """
        Args:
            patience: epochs to wait after last improvement
            verbose: print message on improvement
            delta: minimum change to qualify as improvement
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float, model):
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score - self.delta:
            self.counter += 1
            if self.counter % 25 == 0:
                print(f'\033[1;31mEarlyStopping counter: {self.counter} / {self.patience}\033[0m')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f'\033[1;31mScore improved ({self.best_score:.6f} → {score:.6f})\033[0m')
            self.best_score = score
            self.counter = 0
