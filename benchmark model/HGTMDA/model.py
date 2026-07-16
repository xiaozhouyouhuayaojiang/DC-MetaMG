import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor
from torch.utils.data import DataLoader
from torch_geometric.nn import Linear, GINConv
from torch_geometric.utils import add_self_loops, negative_sampling, degree
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import calculate_metrics
import numpy as np

class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GINConv(Linear(in_channels, hidden_channels), train_eps=True))
        self.convs.append(GINConv(Linear(hidden_channels, out_channels), train_eps=True))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.bns.append(nn.BatchNorm1d(out_channels))

        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ELU()

    def forward(self, x, edge_index):
        edge_index = SparseTensor.from_edge_index(edge_index, sparse_sizes=(x.size(0), x.size(0))).cuda()

        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)

        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = self.activation(x)
        return x


class EdgeDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(EdgeDecoder, self).__init__()
        self.mlps = nn.ModuleList()
        self.mlps.append(nn.Linear(in_channels, hidden_channels))
        self.mlps.append(nn.Linear(hidden_channels, out_channels))

        self.mlps1 = nn.ModuleList()
        self.mlps1.append(nn.Linear(in_channels, hidden_channels))
        self.mlps1.append(nn.Linear(hidden_channels, out_channels))


        self.w = nn.Parameter(torch.tensor(1.0))

        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, z1, edge):
        x = z[edge[0]] * z[edge[1]]
        x1 = z1[edge[0]] * z1[edge[1]]

        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)
        for i, mlp in enumerate(self.mlps1[:-1]):
            x1 = self.dropout(x1)
            x1 = mlp(x1)
            x1 = self.activation(x1)

        x = self.mlps[-1](x)
        x = self.sigmoid(x)
        x1 = self.mlps1[-1](x1)
        x1 = self.sigmoid(x1)
        output = (x + self.w * x1) / (1 + self.w)

        return output


class DegreeDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super(DegreeDecoder, self).__init__()
        self.mlps = nn.ModuleList()
        self.mlps.append(nn.Linear(in_channels, hidden_channels))
        self.mlps.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ELU()

    def forward(self, x):
        for i, mlp in enumerate(self.mlps[:-1]):
            x = mlp(x)
            x = self.dropout(x)
            x = self.activation(x)
        x = self.mlps[-1](x)
        x = self.activation(x)
        return x


def ce_loss(pos_out, neg_out):
    pos_loss = F.binary_cross_entropy(pos_out, torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy(neg_out, torch.zeros_like(neg_out))
    return pos_loss + neg_loss


class GMAE(nn.Module):
    def __init__(self, args, edge_decoder, degree_decoder, mask):
        super(GMAE, self).__init__()
        self.encoder_Coa = GNNEncoder(in_channels=args.dim, hidden_channels=64, out_channels=128)
        self.encoder_fine = GNNEncoder(in_channels=args.dim, hidden_channels=64, out_channels=128)
        self.edge_decoder = edge_decoder
        self.degree_decoder = degree_decoder
        self.mask = mask
        self.loss_fn = ce_loss
        self.negative_sampler = negative_sampling

        self.Noise_Drug_coa = nn.Parameter(torch.tensor(np.random.randn(60, 198)).float().cuda())
        self.Noise_NcRNA_coa= nn.Parameter(torch.tensor(np.random.randn(561, 198)).float().cuda())

        self.Noise_Drug_fine = nn.Parameter(torch.tensor(np.random.randn(60, 198)).float().cuda())
        self.Noise_NcRNA_fine = nn.Parameter(torch.tensor(np.random.randn(561, 198)).float().cuda())

    def train_epoch(self, data, optimizer, alpha, batch_size=8192, grad_norm=1.0):
        x, edge_index = data.X, data.edge_index
        remaining_edges, masked_edges = self.mask(edge_index)
        aug_edge_index, _ = add_self_loops(edge_index)
        neg_edges = self.negative_sampler(
            aug_edge_index, num_nodes=data.num_nodes, num_neg_samples=masked_edges.view(2, -1).size(1)
        ).view_as(masked_edges)
        for perm in DataLoader(range(masked_edges.size(1)), batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()

            x_coa = x[0].detach().clone()
            x_fine = x[1].detach().clone()

            z = self.encoder_Coa(x_coa, remaining_edges)
            z1 = self.encoder_fine(x_fine, remaining_edges)

            batch_masked_edges = masked_edges[:, perm]
            batch_neg_edges = neg_edges[:, perm]
            pos_out = self.edge_decoder(z, z1, batch_masked_edges)
            neg_out = self.edge_decoder(z, z1, batch_neg_edges)
            loss = self.loss_fn(pos_out, neg_out)


            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), grad_norm)
            optimizer.step()

    @torch.no_grad()
    def batch_predict(self, z,z1, edges, batch_size=2 ** 16):
        preds = []
        for perm in DataLoader(range(edges.size(1)), batch_size):
            edge = edges[:, perm]
            preds += [self.edge_decoder(z, z1, edge).squeeze().cpu()]
        pred = torch.cat(preds, dim=0)
        return pred

    @torch.no_grad()
    def test(self, z,z1, pos_edge_index, neg_edge_index):
        pos_pred = self.batch_predict(z,z1, pos_edge_index)
        neg_pred = self.batch_predict(z, z1,neg_edge_index)

        pred = torch.cat([pos_pred, neg_pred], dim=0)
        pos_y = pos_pred.new_ones(pos_pred.size(0))
        neg_y = neg_pred.new_zeros(neg_pred.size(0))

        y = torch.cat([pos_y, neg_y], dim=0)
        y, pred = y.cpu().numpy(), pred.cpu().numpy()

        auc = roc_auc_score(y, pred)
        ap = average_precision_score(y, pred)
        temp = torch.tensor(pred)
        temp[temp >= 0.5] = 1
        temp[temp < 0.5] = 0
        acc, sen, pre, spe, F1, mcc = calculate_metrics(y, temp.cpu())
        return auc, ap, acc, sen, pre, spe, F1, mcc


