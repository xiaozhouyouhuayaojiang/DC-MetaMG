import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from torch.utils.checkpoint import checkpoint

class HetConv(nn.Module):
    def __init__(self, nodes, edges, num_hidden, activation=F.elu, batch_norm=True, negative_slope=0.02):
        super(HetConv, self).__init__()

        self.nodes = nodes
        self.edges = torch.arange(0, edges.max() + 1).to(nodes.device)
        # edge_dim = 16
        edge_dim = num_hidden
        self.edge_embedding = nn.Embedding(edges.max() + 1, edge_dim)

        if batch_norm:
            # self.bn = nn.BatchNorm1d(num_hidden)
            self.bn = nn.Sequential(
                nn.Linear(num_hidden, num_hidden),
                nn.BatchNorm1d(num_hidden)
            )
        else:
            self.bn = None
        self.activation = activation
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.nodes_fc = nn.Parameter(torch.FloatTensor(size=(nodes[:, 1].max() + 1, num_hidden)))
        # self.nodes_fc = nn.Parameter(torch.FloatTensor(size=(1, num_hidden)))
        # self.nodes_fc = self.nodes_fc[self.nodes[:, 1]]
        self.edges_fc = nn.Parameter(torch.FloatTensor(size=(edges.max() + 1, edge_dim)))
        self.nodes_attn = nn.Parameter(torch.FloatTensor(size=(1, num_hidden)))
        self.edges_attn = nn.Parameter(torch.FloatTensor(size=(1, edge_dim)))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.edge_embedding.weight)
        nn.init.xavier_uniform_(self.nodes_fc)
        nn.init.xavier_uniform_(self.edges_fc)
        nn.init.xavier_uniform_(self.nodes_attn)
        nn.init.xavier_uniform_(self.edges_attn)

    def forward(self, g, nodes_feat, edges_feat):
        g = g.local_var()
        nodes_feat = nodes_feat
        # nodes_feat = nodes_feat * self.nodes_fc[self.nodes[:, 1]]
        g.ndata.update({'feat': nodes_feat,
                        'ft': (nodes_feat).sum(dim=-1)})
        g.apply_edges(fn.u_add_v('ft', 'ft', 'e'))
        '''fn.u_add_v 是一个内置的消息函数，它的作用是将源节点（u）和目标节点（v）的特征相加。
        在这个特定的例子中，它将源节点和目标节点的 'ft' 特征相加，生成一个新的消息特征 'e'。这个操作对于图中的每一条边都会执行，结果会存储在边数据（edata）中，键为 'e'
        '''
        all_edge_emb = self.edge_embedding(self.edges)

        ee = (all_edge_emb ).sum(dim=-1)[edges_feat]
        g.edata.update({'ee': ee})

        e = self.leaky_relu(g.edata.pop('e') + g.edata.pop('ee'))
        # g.edata.update({'a': edge_softmax(g, e)})
        g.edata.update({'a': e})

        # message passing
        g.update_all(fn.u_mul_e('feat', 'a', 'm'), fn.sum('m', 'feat'))
        nodes_feat = g.ndata['feat']
        '''
        消息产生：fn.u_mul_e('feat', 'a', 'm') 是一个消息函数，它将节点特征 'feat'（即每个节点的原始特征）与边特征 'a'（即之前计算的边特征，
        表示为边的注意力分数）相乘，结果存储在中间消息 'm' 中。这个操作对于图中的每一条边都会执行，计算出从源节点到目标节点的消息。
        消息聚合：fn.sum('m', 'feat') 是一个聚合函数，它将所有进入每个节点的消息 'm' 进行求和，并将结果更新到目标节点的 'feat' 特征中。这意味着每个节点的新特征将是所有其邻居节点发送的消息之和。
        '''
        if self.bn:
            nodes_feat = self.bn(nodes_feat)
        if self.activation:
            nodes_feat = self.activation(nodes_feat)
        return nodes_feat

class HGNN(nn.Module):
    def __init__(self,
                 g,
                 edges,
                 nodes,
                 num_hidden,
                 num_layer=3
                 ):
        super(HGNN, self).__init__()
        self.g = g
        self.nodes = nodes
        self.edges = edges

        self.num_layer = num_layer
        self.num_hidden = num_hidden

        self.residual = True
        self.dropout = nn.Dropout(0.2)
        # self.dropout = nn.Identity()
        self.bn = True

        self.node_embedding = nn.Embedding(self.nodes[:, 0].max() + 1, num_hidden)

        self.init_node_features_by_type()

        self.Linear = nn.Linear(5,num_hidden)

        self.gat_layers = nn.ModuleList()
        for l in range(self.num_layer):
            self.gat_layers.append(HetConv(nodes, edges, num_hidden, activation=F.relu, batch_norm=self.bn))
        # self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_embedding.weight)

    def init_emb(self, emb):
        self.node_embedding = self.node_embedding.from_pretrained(emb.float(), freeze=True)

    def init_node_features_by_type(self):
        """
        Initialize node features such that nodes of the same type have the same features.
        Returns:
            torch.Tensor: A tensor of shape [num_nodes, num_hidden] containing initial features.
        """
        # Get unique node types
        node_types = self.nodes[:, 1].unique()
        num_types = len(node_types)

        type_features = torch.randn(num_types, self.num_hidden)

        # Assign features to each node based on its type
        node_features = torch.zeros(len(self.nodes), self.num_hidden)
        for node_idx, (_, node_type) in enumerate(self.nodes):
            type_idx = (node_types == node_type).nonzero().item()
            node_features[node_idx] = type_features[type_idx]
        self.init_emb(node_features)

        return node_features

    def get_output_size(self):
        # return (self.num_layer + 1) * self.num_hidden
        return self.num_hidden

    def forward(self):
        '''
        all_layer_node_feats = [self.node_embedding(self.nodes[:, 0])]

        '''
        all_layer_node_3Dfeats = self.Linear(self.g.ndata['3D'])
        all_layer_node_feats = self.node_embedding(self.nodes[:, 0])
        all_layer_node_feats = all_layer_node_3Dfeats + all_layer_node_feats
        all_layer_node_feats = [all_layer_node_feats]

        for l in range(self.num_layer):
            node_feats = self.gat_layers[l](self.g, all_layer_node_feats[-1], self.edges)
            if self.residual:
                #node_feats = node_feats + all_layer_node_feats[-1]
                node_feats = node_feats + all_layer_node_feats[0]
            node_feats = self.dropout(node_feats)
            all_layer_node_feats.append(node_feats)

        return all_layer_node_feats[-1]
