from itertools import combinations
import torch.nn.functional as F
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
from torch import nn, Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_undirected, sort_edge_index, degree
from torch_geometric.utils.num_nodes import maybe_num_nodes
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
from torch_geometric.data import HeteroData


class GCN_GAT_GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        self.gcn1 = GCNConv(in_channels, hidden_channels)

        self.gat = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False, edge_dim=1)

        self.gcn2 = GCNConv(hidden_channels, out_channels)

        # cnn
        self.cnn = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)

    def forward(self, x, edge_index):

        x1 = self.gcn1(x, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        x2 = self.gat(x1, edge_index)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x3 = self.gcn2(x2, edge_index)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, p=self.dropout, training=self.training)


        x = torch.cat((x1, x3), dim=1)

        x = x.T.unsqueeze(0)

        x = self.cnn(x)
        x = x.squeeze(0).T

        print(x.shape)
        return x


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def draw_auc(y, pred, l):
    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='{}:AUC = %0.4f'.format(l) % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)


def draw_aupr(y, pred, l):
    average_precision = average_precision_score(y, pred)
    precision, recall, _ = precision_recall_curve(y, pred)
    plt.plot(recall, precision, label='{}:AUPR = %0.4f'.format(l) % average_precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUPR Curve')
    plt.legend(loc='lower right')
    plt.grid(True)


def print_result(result):
    metrics = ['auc', 'aupr', 'acc', 'sen', 'pre', 'spe', 'F1', 'mcc']
    metric_values = [[] for _ in range(len(metrics))]
    for i in result:
        for j, val in enumerate(i):
            metric_values[j].append(val)
    metric_values = [np.array(m) for m in metric_values]
    formatted_metrics = []
    for metric, values in zip(metrics, metric_values):
        mean = "{:.4f}".format(values.mean())
        std = "{:.4f}".format(np.std(values))
        formatted_metrics.append(f"{metric}: {mean} ± {std}")
    print(*formatted_metrics)
    return formatted_metrics

def mask_path(edge_index, p, walks_per_node, walk_length, num_nodes):

    edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    edge_index = sort_edge_index(edge_index, num_nodes=num_nodes)
    row, col = edge_index

    sample_mask = torch.rand(row.size(0), device=edge_index.device) <= p
    start = row[sample_mask].repeat(walks_per_node)

    deg = degree(row, num_nodes=num_nodes)
    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])

    n_id, e_id = torch.ops.torch_cluster.random_walk(rowptr, col, start, walk_length, 1.0, 1.0)

    e_id = e_id[e_id != -1].view(-1)

    edge_mask[e_id] = False

    return edge_index[:, edge_mask], edge_index[:, ~edge_mask]


class MaskPath(torch.nn.Module):
    def __init__(self, p, walk_length, num_nodes):
        super(MaskPath, self).__init__()
        self.p = p
        self.walk_length = walk_length
        self.num_nodes = num_nodes

    def forward(self, edge_index):
        remaining_edges, masked_edges = mask_path(edge_index, self.p, 1, self.walk_length, self.num_nodes)
        remaining_edges = to_undirected(remaining_edges)
        return remaining_edges, masked_edges


def calculate_metrics(y_true, y_pred):
    TP = sum((y_true[i] == 1 and y_pred[i] == 1) for i in range(len(y_true)))
    TN = sum((y_true[i] == 0 and y_pred[i] == 0) for i in range(len(y_true)))
    FP = sum((y_true[i] == 0 and y_pred[i] == 1) for i in range(len(y_true)))
    FN = sum((y_true[i] == 1 and y_pred[i] == 0) for i in range(len(y_true)))

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-10)
    sensitivity = TP / (TP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    specificity = TN / (TN + FP + 1e-10)
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    F1_score = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-10)
    return accuracy, sensitivity, precision, specificity, F1_score, mcc


def get_data(type):
    print("Loading data.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if type != "miRNA":
        drug_feat_fine = np.loadtxt("data/miRNA/drug_feature.txt")          # (60,  198)
        mirna_feat_fine = np.loadtxt("data/miRNA/Feature_MiRNA_SQ.txt")     # (561, 198)

        drug_feat_coarse = np.loadtxt("data/miRNA/GIP_Topo_drug.txt")       # (60,  198)
        mirna_feat_coarse = np.loadtxt("data/miRNA/Gaussian_sim_MiRNA.txt") # (561, 198)

        edges_np = np.loadtxt(f"data/miRNA/edges.txt", dtype=np.int64)
    else:
        drug_feat_fine = np.loadtxt("data/lncRNA/drug_feature.txt")  # (60,  198)
        mirna_feat_fine = np.loadtxt("data/lncRNA/Feature_LncRNA_Sq.txt")  # (561, 198)

        drug_feat_coarse = np.loadtxt("data/lncRNA/GIP_Topo_drug.txt")  # (60,  198)
        mirna_feat_coarse = np.loadtxt("data/lncRNA/Gaussian_sim_LncRNA.txt")  # (561, 198)

        edges_np = np.loadtxt(f"data/lncRNA/edges.txt", dtype=np.int64)


    drug_features = torch.tensor(
        np.concatenate([drug_feat_fine, drug_feat_coarse], axis=1),
        dtype=torch.float32)                                       # (60,  396)
    mirna_features = torch.tensor(
        np.concatenate([mirna_feat_fine, mirna_feat_coarse], axis=1),
        dtype=torch.float32)                                       # (561, 396)

    num_drugs = drug_features.size(0)   # 60
    num_miRNA = mirna_features.size(0)  # 561

    # edges.txt: col0=drug_index, col1=mirna_index

    drug_idx  = edges_np[:, 0]   # 0-based drug index
    mirna_idx = edges_np[:, 1]   # 0-based mirna index

    src = torch.tensor(mirna_idx, dtype=torch.long)
    dst = torch.tensor(drug_idx + num_miRNA, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)   # [2, num_edges]

    num_nodes = num_miRNA + num_drugs  # 621

    combined_features = torch.cat([mirna_features, drug_features], dim=0)  # (621, 396)

    data_pyg = Data(
        x=combined_features,
        edge_index=edge_index,
        num_nodes=num_nodes,
    )

    transform = T.RandomLinkSplit(
        num_val=0,
        num_test=0.2,
        is_undirected=True,
        split_labels=True,
        add_negative_train_samples=True,
    )
    train_data, _, test_data = transform(data_pyg)

    train_data = train_data.to(device)
    test_data  = test_data.to(device)

    splits = dict(train=train_data, test=test_data)
    return splits


def fully_connected_edge_index(num_nodes):
    edge_index = torch.tensor(list(combinations(range(num_nodes), 2)), dtype=torch.long).T

    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    return edge_index
