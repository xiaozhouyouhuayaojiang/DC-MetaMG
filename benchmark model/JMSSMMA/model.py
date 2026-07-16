import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv,GCNConv


class DrugLncRNAGNN(nn.Module):


    def __init__(self, drug_feat_dim=198, lncrna_feat_dim=198,
                 in_channels=256, hidden_channels=128, num_layers=2, dropout=0.3,
                 n_drug=60):
        super(DrugLncRNAGNN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.n_drug = n_drug  # number of drug nodes (prefix of x)

        self.drug_proj   = nn.Linear(drug_feat_dim,   in_channels)
        self.lncrna_proj = nn.Linear(lncrna_feat_dim, in_channels)

        # GNN layers for bipartite graph
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            self.convs.append(
                SAGEConv((in_dim, in_dim), hidden_channels)
            )

    def forward(self, x, edge_index):

        # Apply separate projection for drug and lncRNA nodes
        x_drug   = F.relu(self.drug_proj(x[:self.n_drug]))
        x_ncrna  = F.relu(self.lncrna_proj(x[self.n_drug:]))
        x = torch.cat([x_drug, x_ncrna], dim=0)

        # Message passing
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(F.dropout(x, p=self.dropout, training=self.training))

        return x


class LinkPredictor(nn.Module):


    def __init__(self, hidden_channels=128, dropout=0.3):
        super(LinkPredictor, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, z, edge_label_index):

        # Extract embeddings for edge endpoints

        drug_emb = z[edge_label_index[0]]
        lncrna_emb = z[edge_label_index[1]]
        edge_emb = drug_emb * lncrna_emb
        return self.mlp(edge_emb).squeeze()


class DrugLncRNAPredictor(nn.Module):


    def __init__(self, drug_feat_dim=198, lncrna_feat_dim=198,
                 in_channels=256, hidden_channels=128, num_layers=2, dropout=0.3,
                 n_drug=60):
        super(DrugLncRNAPredictor, self).__init__()

        self.gnn = DrugLncRNAGNN(drug_feat_dim, lncrna_feat_dim,
                                  in_channels, hidden_channels, num_layers,
                                  dropout, n_drug)
        self.link_pred = LinkPredictor(hidden_channels, dropout)

    def encode(self, x, edge_index):
        return self.gnn(x, edge_index)

    def decode(self, z, edge_label_index):
        return self.link_pred(z, edge_label_index)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)
