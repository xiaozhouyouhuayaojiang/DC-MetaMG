import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import softmax


class HypergraphConv(nn.Module):
    """Hypergraph convolution layer"""
    def __init__(self, in_channels, out_channels):
        super(HypergraphConv, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, hyperedge_index, D_v, D_e):

        num_nodes = x.shape[0]
        num_hyperedges = int(hyperedge_index[1].max().item() + 1)

        # Build incidence matrix H: [num_nodes, num_hyperedges]
        H = torch.zeros(num_nodes, num_hyperedges, device=x.device)
        H[hyperedge_index[0], hyperedge_index[1]] = 1.0

        # D_v^(-1/2)
        D_v_inv_sqrt = torch.pow(D_v + 1e-10, -0.5)
        D_v_inv_sqrt = torch.diag(D_v_inv_sqrt).to(x.device)

        # D_e^(-1)
        D_e_inv = torch.pow(D_e + 1e-10, -1.0)
        D_e_inv = torch.diag(D_e_inv).to(x.device)

        # Apply transformation: D_v^(-1/2) @ H @ D_e^(-1) @ H^T @ D_v^(-1/2) @ X
        x_transformed = D_v_inv_sqrt @ H @ D_e_inv @ H.T @ D_v_inv_sqrt @ x

        # Linear transformation
        out = self.linear(x_transformed)

        return out


class LowOrderGraphEncoder(nn.Module):

    def __init__(self, in_channels, hidden_channels, num_layers=6, dropout=0.5):
        super(LowOrderGraphEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

    def forward(self, x, edge_index):

        for i in range(self.num_layers):
            x_new = self.convs[i](x, edge_index)
            x_new = self.batch_norms[i](x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)

            # Residual connection (if dimensions match)
            if x.shape[1] == x_new.shape[1]:
                x = x + x_new
            else:
                x = x_new

        return x


class HighOrderHypergraphEncoder(nn.Module):

    def __init__(self, in_channels, hidden_channels, num_layers=3, dropout=0.5):
        super(HighOrderHypergraphEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        self.convs.append(HypergraphConv(in_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(HypergraphConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

    def forward(self, x, hypergraph_data):

        hyperedge_index = hypergraph_data['hyperedge_index']
        D_v = hypergraph_data['D_v']
        D_e = hypergraph_data['D_e']

        for i in range(self.num_layers):
            x = self.convs[i](x, hyperedge_index, D_v, D_e)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class HeterogeneousInteractionEncoder(nn.Module):

    def __init__(self, in_channels, hidden_channels, num_edge_types=3, dropout=0.5):
        super(HeterogeneousInteractionEncoder, self).__init__()
        self.num_edge_types = num_edge_types
        self.dropout = dropout

        # Type-specific linear transformations
        self.edge_embeddings = nn.ModuleList([
            nn.Linear(in_channels, hidden_channels) for _ in range(num_edge_types)
        ])

        # Aggregation weights
        self.type_weights = nn.Parameter(torch.ones(num_edge_types))

        self.batch_norm = nn.BatchNorm1d(hidden_channels)

    def forward(self, x, hetero_data):
        edge_index = hetero_data['edge_index']
        edge_type = hetero_data['edge_type']
        num_nodes = x.shape[0]

        type_outputs = []
        for t in range(self.num_edge_types):
            mask = (edge_type == t)
            type_edges = edge_index[:, mask]

            if type_edges.shape[1] > 0:
                src, dst = type_edges


                valid_mask = (src >= 0) & (src < num_nodes) & (dst >= 0) & (dst < num_nodes)
                if valid_mask.sum() == 0:
                    type_outputs.append(torch.zeros(num_nodes, self.hidden_channels, device=x.device))
                    continue

                src = src[valid_mask]
                dst = dst[valid_mask]


                x_transformed = self.edge_embeddings[t](x)
                messages = x_transformed[src]

                out = torch.zeros(num_nodes, x_transformed.shape[1], device=x.device)
                out.index_add_(0, dst, messages)

                degree = torch.zeros(num_nodes, device=x.device)
                degree.index_add_(0, dst, torch.ones(dst.shape[0], device=x.device))
                degree = degree.clamp(min=1).unsqueeze(1)
                out = out / degree

                type_outputs.append(out)
            else:

                type_outputs.append(torch.zeros(num_nodes, self.hidden_channels, device=x.device))



        weights = F.softmax(self.type_weights, dim=0)

        output = sum(w * out for w, out in zip(weights, type_outputs))

        output = self.batch_norm(output)
        output = F.relu(output)
        output = F.dropout(output, p=self.dropout, training=self.training)

        return output


class BiasedTransformerFusion(nn.Module):

    def __init__(self, hidden_channels, num_heads=8, num_layers=2, dropout=0.1):
        super(BiasedTransformerFusion, self).__init__()

        # Learnable channel embeddings (3 channels)
        self.channel_embeddings = nn.Parameter(torch.randn(3, hidden_channels))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=num_heads,
            dim_feedforward=hidden_channels * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(hidden_channels)

    def forward(self, x_stack):

        # Add channel-specific biases
        x_biased = x_stack + self.channel_embeddings.unsqueeze(0)

        # Apply transformer
        x_transformed = self.transformer(x_biased)

        # Mean pooling across channels
        x_fused = x_transformed.mean(dim=1)

        # Layer normalization
        x_fused = self.layer_norm(x_fused)

        return x_fused


class GCNRefine(nn.Module):

    def __init__(self, in_channels, out_channels, num_layers=6, dropout=0.5):
        super(GCNRefine, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Calculate intermediate dimensions
        hidden_channels = in_channels

        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Final layer to output dimension
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.batch_norms.append(nn.BatchNorm1d(out_channels))

        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):

        x_ = self.linear(x)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            if i < self.num_layers - 1:  # No activation on last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + x_
        return x


class TriFusion(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(TriFusion, self).__init__()

        # Three encoders
        self.low_order_encoder = LowOrderGraphEncoder(
            in_channels, hidden_channels, num_layers=6, dropout=dropout
        )
        self.high_order_encoder = HighOrderHypergraphEncoder(
            in_channels, hidden_channels, num_layers=3, dropout=dropout
        )
        self.hetero_encoder = HeterogeneousInteractionEncoder(
            in_channels, hidden_channels, num_edge_types=3, dropout=dropout
        )

        # Fusion encoder
        self.fusion = BiasedTransformerFusion(
            hidden_channels, num_heads=8, num_layers=2, dropout=dropout
        )

        # GCN refinement
        self.gcn_refine = GCNRefine(
            hidden_channels, out_channels, num_layers=6, dropout=dropout
        )

        # MLP decoder for link prediction
        self.decoder = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels // 2, 1)
        )

    def forward(self, x, knn_edge_index, hypergraph_data, hetero_data):

        # Encode three channels
        h_low = self.low_order_encoder(x, knn_edge_index)
        h_high = self.high_order_encoder(x, hypergraph_data)
        h_hetero = self.hetero_encoder(x, hetero_data)

        # Stack and fuse
        h_stack = torch.stack([h_low, h_high, h_hetero], dim=1)  # [num_nodes, 3, hidden]
        h_fused = self.fusion(h_stack)  # [num_nodes, hidden]

        # Refine with GCN
        z = self.gcn_refine(h_fused, knn_edge_index)  # [num_nodes, out_channels]

        return z

    def decode(self, z, edge_index):

        src, dst = edge_index
        z_src = z[src]
        z_dst = z[dst]

        # Hadamard product (element-wise multiplication)
        edge_feat = z_src * z_dst

        # MLP decoder
        out = self.decoder(edge_feat).squeeze()

        return out
