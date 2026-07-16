

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        return output


class MHGCN(nn.Module):

    def __init__(self, drug_in_dim, ncrna_in_dim, nfeat, nhid, out, dropout=0.0, num_layers=1):
        super(MHGCN, self).__init__()

        self.nfeat = nfeat
        self.out = out
        self.dropout = dropout
        self.num_layers = num_layers

        # Trainable type-specific feature projections (fix: was frozen in utils.py)
        self.drug_proj = nn.Linear(drug_in_dim, nfeat, bias=False)
        self.ncrna_proj = nn.Linear(ncrna_in_dim, nfeat, bias=False)
        nn.init.xavier_uniform_(self.drug_proj.weight)
        nn.init.xavier_uniform_(self.ncrna_proj.weight)

        # Graph convolution layers
        self.gc1 = GraphConvolution(nfeat, out)
        self.gc2 = GraphConvolution(out, out)
        self.gc3 = GraphConvolution(out, out)
        self.gc4 = GraphConvolution(out, out)
        self.gc5 = GraphConvolution(out, out)

        # BatchNorm after each layer
        self.bn1 = nn.BatchNorm1d(out)
        self.bn2 = nn.BatchNorm1d(out)
        self.bn3 = nn.BatchNorm1d(out)
        self.bn4 = nn.BatchNorm1d(out)
        self.bn5 = nn.BatchNorm1d(out)

        # Learnable scalar weight for bipartite interaction graph
        self.weight_bipartite = nn.Parameter(torch.FloatTensor(1))

        # Learnable layer fusion weights (softmax-normalized in forward)
        self.weight_l1 = nn.Parameter(torch.FloatTensor(1))
        self.weight_l2 = nn.Parameter(torch.FloatTensor(1))
        self.weight_l3 = nn.Parameter(torch.FloatTensor(1))
        self.weight_l4 = nn.Parameter(torch.FloatTensor(1))
        self.weight_l5 = nn.Parameter(torch.FloatTensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        self.weight_bipartite.data.uniform_(0.08, 0.12)
        for w in [self.weight_l1, self.weight_l2, self.weight_l3,
                  self.weight_l4, self.weight_l5]:
            nn.init.uniform_(w)

    def _apply_layer(self, gc, bn, h, adj):
        return F.dropout(bn(F.relu(gc(h, adj))), self.dropout, training=self.training)

    def forward(self, drug_raw, ncrna_raw, adj_bipartite, num_drugs, num_ncrnas):

        device = drug_raw.device

        # Trainable type-specific projections
        drug_feat = self.drug_proj(drug_raw)
        ncrna_feat = self.ncrna_proj(ncrna_raw)
        features = torch.cat([drug_feat, ncrna_feat], dim=0)  # (N, nfeat)

        # Scale bipartite adjacency with learnable weight
        bi_idx = adj_bipartite.coalesce().indices()
        bi_val = adj_bipartite.coalesce().values() * self.weight_bipartite
        N = num_drugs + num_ncrnas
        adj_combined = torch.sparse.FloatTensor(bi_idx, bi_val, torch.Size([N, N])).to(device)

        h1 = self._apply_layer(self.gc1, self.bn1, features, adj_combined)
        h2 = self._apply_layer(self.gc2, self.bn2, h1, adj_combined)

        layer_outputs = [h1, h2]
        raw_ws = [self.weight_l1, self.weight_l2]

        if self.num_layers >= 3:
            h3 = self._apply_layer(self.gc3, self.bn3, h2, adj_combined)
            layer_outputs.append(h3)
            raw_ws.append(self.weight_l3)

        if self.num_layers >= 4:
            h4 = self._apply_layer(self.gc4, self.bn4, h3, adj_combined)
            layer_outputs.append(h4)
            raw_ws.append(self.weight_l4)

        if self.num_layers >= 5:
            h5 = self._apply_layer(self.gc5, self.bn5, h4, adj_combined)
            layer_outputs.append(h5)
            raw_ws.append(self.weight_l5)

        norm_ws = F.softmax(torch.stack(raw_ws[:len(layer_outputs)]).squeeze(-1), dim=0)
        output = sum(h * w for h, w in zip(layer_outputs, norm_ws))

        return output


class LinkPredictor(nn.Module):

    def __init__(self, in_dim, dropout=0.3):
        super(LinkPredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, 1)
        )

    def forward(self, emb_src, emb_dst):

        x = torch.cat([emb_src, emb_dst], dim=-1)
        return self.mlp(x).squeeze(-1)  # (batch,)
