import numpy as np
import torch
import random
from torch_geometric.data import Data
import torch_geometric.transforms as T
from collections import deque
import pandas as pd
from torch_geometric.data import HeteroData
import os

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_trifusion_data(data_path):
    """
    Load LncRNA-Drug data with features and edges

    Args:
        data_path: Path to data directory

    Returns:
        train_data: Training data split
        test_data: Test data split
        num_lncrnas: Number of LncRNA nodes
        num_drugs: Number of Drug nodes
    """
    # Load LncRNA features (396 dimensions)
    lncrna_features = []
    with open(f'{data_path}/Feature_{data_path}_Sq.txt', 'r') as f:
        for line in f:
            features = [float(x) for x in line.strip().split()]
            lncrna_features.append(features)
    lncrna_features = np.array(lncrna_features)

    # Load Drug features (396 dimensions)
    drug_features = []
    with open(f'{data_path}/drug_feature.txt', 'r') as f:
        for line in f:
            features = [float(x) for x in line.strip().split()]
            drug_features.append(features)
    drug_features = np.array(drug_features)

    num_lncrnas = lncrna_features.shape[0]
    num_drugs = drug_features.shape[0]

    # Concatenate features: [lncrna_features; drug_features]
    all_features = np.vstack([lncrna_features, drug_features])
    x = torch.FloatTensor(all_features)

    '''rel_dedup = pd.read_csv(f'{data_path}/edges.txt')
    drug_ei, rna_ei = rel_dedup['Drug_Index'].values, rel_dedup['Protein_Index'].values
    data = HeteroData()

    data['drug'].x = torch.tensor(drug_x, dtype=torch.float32)
    data['protein'].x = torch.tensor(prot_x, dtype=torch.float32)

    edge_index = torch.tensor(
        np.stack([drug_ei, prot_ei], axis=0), dtype=torch.long)  # (2, E)
    data['drug', 'interacts', 'protein'].edge_index = edge_index

    print(f'Graph: {data["drug"].x.shape[0]} drugs, '
          f'{data["protein"].x.shape[0]} proteins, '
          f'{edge_index.shape[1]} positive pairs')'''
    # Load edges (lncrna_idx, drug_idx)
    edges = []
    with open(f'{data_path}/edges.txt', 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                lncrna_idx = int(parts[1])
                drug_idx = int(parts[0])
                # Offset drug index by num_lncrnas to create global indexing
                edges.append([lncrna_idx, drug_idx + num_lncrnas])

    edges = np.array(edges).T
    edge_index = torch.LongTensor(edges)

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index)

    # Apply RandomLinkSplit as specified
    transform = T.RandomLinkSplit(
        num_val=0,
        num_test=0.2,
        is_undirected=True,
        split_labels=True,
        add_negative_train_samples=True
    )

    if os.path.exists(f'{data_path}/splits.pt'):
        transform = torch.load(f"{data_path}/splits.pt")
    else:
        save_path = f"{data_path}/splits.pt"
        torch.save(transform, save_path)

    train_data, _, test_data = transform(data)
    print(f"Data loaded: {num_lncrnas} LncRNAs, {num_drugs} Drugs")
    print(f"Total nodes: {x.shape[0]}, Feature dim: {x.shape[1]}")
    print(f"Train edges: {train_data.pos_edge_label_index.shape[1]}")
    print(f"Test edges: {test_data.pos_edge_label_index.shape[1]}")



    return train_data, test_data, num_lncrnas, num_drugs


def build_knn_graph(similarity_matrix, k=40):
    """
    Build K-nearest neighbor graph from similarity matrix

    Args:
        similarity_matrix: (n, n) numpy array of similarities
        k: number of nearest neighbors

    Returns:
        edge_index: [2, num_edges] tensor
    """
    sim_tensor = torch.FloatTensor(similarity_matrix)
    n = sim_tensor.shape[0]

    # For each node, find k nearest neighbors
    edges = []
    for i in range(n):
        # Get similarities for node i
        sims = sim_tensor[i]
        # Find top-k neighbors (excluding self)
        _, top_k_indices = torch.topk(sims, k + 1)
        # Remove self-loop
        top_k_indices = top_k_indices[top_k_indices != i][:k]

        # Add bidirectional edges
        for j in top_k_indices:
            edges.append([i, j.item()])

    edges = torch.LongTensor(edges).T
    print(f"KNN graph built: {edges.shape[1]} edges for {n} nodes (K={k})")

    return edges


def build_hypergraph(knn_edge_index, num_nodes, N=3):
    """
    Build N-order hypergraph from K-NN graph using BFS

    Args:
        knn_edge_index: [2, num_edges] base graph edges
        num_nodes: total number of nodes
        N: order of neighborhood (default 3)

    Returns:
        hyperedge_index: [2, num_hyperedges] node-to-hyperedge mapping
        D_v: node degree diagonal matrix
        D_e: hyperedge degree diagonal matrix
    """
    # Build adjacency list for BFS
    adj_list = [[] for _ in range(num_nodes)]
    for i in range(knn_edge_index.shape[1]):
        src, dst = knn_edge_index[0, i].item(), knn_edge_index[1, i].item()
        adj_list[src].append(dst)

    # For each node, collect N-hop neighbors using BFS
    hyperedges = []
    for start_node in range(num_nodes):
        visited = set()
        queue = deque([(start_node, 0)])
        visited.add(start_node)
        n_hop_neighbors = [start_node]

        while queue:
            node, depth = queue.popleft()
            if depth < N:
                for neighbor in adj_list[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        n_hop_neighbors.append(neighbor)
                        queue.append((neighbor, depth + 1))

        # Create hyperedge from this node and its N-hop neighbors
        if len(n_hop_neighbors) > 1:
            hyperedges.append(n_hop_neighbors)

    # Build incidence matrix representation
    # hyperedge_index[0] = node indices, hyperedge_index[1] = hyperedge indices
    node_indices = []
    hyperedge_indices = []
    for he_idx, he_nodes in enumerate(hyperedges):
        for node_idx in he_nodes:
            node_indices.append(node_idx)
            hyperedge_indices.append(he_idx)

    hyperedge_index = torch.LongTensor([node_indices, hyperedge_indices])

    # Compute degree matrices
    # D_v: node degree (number of hyperedges each node belongs to)
    D_v = torch.zeros(num_nodes)
    for node_idx in node_indices:
        D_v[node_idx] += 1

    # D_e: hyperedge degree (number of nodes in each hyperedge)
    D_e = torch.zeros(len(hyperedges))
    for he_idx in hyperedge_indices:
        D_e[he_idx] += 1

    print(f"Hypergraph built: {len(hyperedges)} hyperedges, avg size: {D_e.mean().item():.2f}")

    return hyperedge_index, D_v, D_e


def build_heterogeneous_graph(lncrna_knn_edges, drug_knn_edges, interaction_edges, num_lncrnas, num_drugs):
    """
    Build heterogeneous graph with three edge types

    Args:
        lncrna_knn_edges: [2, num_edges] LncRNA-LncRNA edges
        drug_knn_edges: [2, num_edges] Drug-Drug edges
        interaction_edges: [2, num_edges] LncRNA-Drug interaction edges
        num_lncrnas: number of LncRNA nodes
        num_drugs: number of Drug nodes

    Returns:
        edge_index: [2, total_edges] combined edge index
        edge_type: [total_edges] edge type labels (0: LncRNA-LncRNA, 1: Drug-Drug, 2: LncRNA-Drug)
    """
    # Type 0: LncRNA-LncRNA (indices 0 to num_lncrnas-1)
    type_0_edges = lncrna_knn_edges
    type_0_labels = torch.zeros(type_0_edges.shape[1], dtype=torch.long)

    # Type 1: Drug-Drug (indices num_lncrnas to num_lncrnas+num_drugs-1)
    # Drug indices in knn graph are 0-based, need to offset
    type_1_edges = drug_knn_edges + num_lncrnas
    type_1_labels = torch.ones(type_1_edges.shape[1], dtype=torch.long)

    # Type 2: LncRNA-Drug (bipartite, already in global indexing)
    type_2_edges = interaction_edges
    type_2_labels = torch.full((type_2_edges.shape[1],), 2, dtype=torch.long)

    # Combine all edge types
    edge_index = torch.cat([type_0_edges, type_1_edges, type_2_edges], dim=1)
    edge_type = torch.cat([type_0_labels, type_1_labels, type_2_labels])

    print(f"Heterogeneous graph built:")
    print(f"  Type 0 (LncRNA-LncRNA): {type_0_edges.shape[1]} edges")
    print(f"  Type 1 (Drug-Drug): {type_1_edges.shape[1]} edges")
    print(f"  Type 2 (LncRNA-Drug): {type_2_edges.shape[1]} edges")
    print(f"  Total: {edge_index.shape[1]} edges")

    return edge_index, edge_type


def combine_knn_graphs(lncrna_knn_edges, drug_knn_edges, num_lncrnas):
    """
    Combine LncRNA and Drug KNN graphs into single graph with global indexing

    Args:
        lncrna_knn_edges: [2, num_edges] LncRNA KNN edges
        drug_knn_edges: [2, num_edges] Drug KNN edges
        num_lncrnas: number of LncRNA nodes (offset for drug indices)

    Returns:
        combined_edges: [2, total_edges] combined edge index
    """
    # Drug indices need to be offset by num_lncrnas
    drug_knn_offset = drug_knn_edges + num_lncrnas

    # Combine both graphs
    combined_edges = torch.cat([lncrna_knn_edges, drug_knn_offset], dim=1)

    return combined_edges
