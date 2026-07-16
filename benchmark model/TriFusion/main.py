import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, recall_score
import os

from util import set_seed, load_trifusion_data, build_knn_graph, build_hypergraph, build_heterogeneous_graph, combine_knn_graphs
from model import TriFusion

import pandas as pd
from collections import defaultdict

def train_epoch(model, train_data, optimizer, graphs, device):

    model.train()
    optimizer.zero_grad()

    # Forward pass
    z = model(
        train_data.x.to(device),
        graphs['knn'].to(device),
        graphs['hyper'],
        graphs['hetero']
    )

    # Decode positive and negative edges
    pos_pred = model.decode(z, train_data.pos_edge_label_index.to(device))
    neg_pred = model.decode(z, train_data.neg_edge_label_index.to(device))

    # Binary cross-entropy loss
    pos_loss = F.binary_cross_entropy_with_logits(pos_pred, torch.ones_like(pos_pred))
    neg_loss = F.binary_cross_entropy_with_logits(neg_pred, torch.zeros_like(neg_pred))
    loss = pos_loss + neg_loss

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, data, graphs, device):

    model.eval()

    # Forward pass
    z = model(
        data.x.to(device),
        graphs['knn'].to(device),
        graphs['hyper'],
        graphs['hetero']
    )

    # Decode edges
    pos_pred = model.decode(z, data.pos_edge_label_index.to(device)).sigmoid()
    neg_pred = model.decode(z, data.neg_edge_label_index.to(device)).sigmoid()

    # Combine predictions and labels
    preds = torch.cat([pos_pred, neg_pred]).cpu().numpy()
    labels = torch.cat([
        torch.ones(pos_pred.size(0)),
        torch.zeros(neg_pred.size(0))
    ]).numpy()

    # Compute metrics
    auc = roc_auc_score(labels, preds)
    aupr = average_precision_score(labels, preds)

    # Threshold at 0.5 for classification metrics
    preds_binary = (preds >= 0.5).astype(int)
    acc = accuracy_score(labels, preds_binary)
    f1 = f1_score(labels, preds_binary)
    precision = precision_score(labels, preds_binary)
    recall = recall_score(labels, preds_binary)

    return auc, aupr, acc, f1, precision, recall


def main():


    # Configuration
    config = {
        'type':'miRNA',
        'data_path': r'miRNA',
        'hidden_channels': 256,
        'out_channels': 128,
        'k_neighbors': 40,
        'n_order': 3,
        'lr': 0.001,
        'weight_decay': 5e-4,
        'dropout': 0.5,
        'epochs': 200,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42
    }

    print("=" * 80)
    print("TriFusion: Tri-Channel Fusion Neural Network")
    print("=" * 80)
    print(f"Device: {config['device']}")
    print(f"Configuration: {config}")
    print()

    # Set random seed
    set_seed(config['seed'])

    # Load data
    print("Loading data...")
    train_data, test_data, num_lncrnas, num_drugs = load_trifusion_data(config['data_path'])
    print()

    # Load feature vectors and compute similarity matrices
    print("Loading feature vectors and computing similarity matrices...")
    lncrna_features_for_sim = np.loadtxt(f"{config['data_path']}/Gaussian_sim_{config['type']}.txt")
    drug_features_for_sim = np.loadtxt(f"{config['data_path']}/GIP_Topo_drug.txt")
    print(f"LncRNA features shape: {lncrna_features_for_sim.shape}")
    print(f"Drug features shape: {drug_features_for_sim.shape}")

    from sklearn.metrics.pairwise import cosine_similarity
    lncrna_sim = cosine_similarity(lncrna_features_for_sim)

    drug_sim = cosine_similarity(drug_features_for_sim)
    # Compute cosine similarity matrices

    lncrna_sim = np.where(lncrna_sim > 0.9, lncrna_sim, 0)
    drug_sim = np.where(drug_sim > 0.9, drug_sim, 0)

    print(f"LncRNA similarity matrix: {lncrna_sim.shape}")
    print(f"Drug similarity matrix: {drug_sim.shape}")
    print()


    # Build K-NN graphs
    print("Building K-NN graphs...")
    lncrna_knn_edges = build_knn_graph(lncrna_sim, k=config['k_neighbors'])
    drug_knn_edges = build_knn_graph(drug_sim, k=config['k_neighbors'])
    knn_edge_index = combine_knn_graphs(lncrna_knn_edges, drug_knn_edges, num_lncrnas)
    print()

    # Build hypergraphs
    print("Building hypergraphs...")
    lncrna_hyper_index, lncrna_D_v, lncrna_D_e = build_hypergraph(
        lncrna_knn_edges, num_lncrnas, N=config['n_order']
    )
    drug_hyper_index, drug_D_v, drug_D_e = build_hypergraph(
        drug_knn_edges, num_drugs, N=config['n_order']
    )

    # Combine hypergraphs with offset for drug indices
    drug_hyper_index_offset = drug_hyper_index.clone()
    drug_hyper_index_offset[0] += num_lncrnas  # Offset node indices
    drug_hyper_index_offset[1] += int(lncrna_hyper_index[1].max().item() + 1)  # Offset hyperedge indices

    combined_hyper_index = torch.cat([lncrna_hyper_index, drug_hyper_index_offset], dim=1)
    combined_D_v = torch.cat([lncrna_D_v, drug_D_v])
    combined_D_e = torch.cat([lncrna_D_e, drug_D_e])

    print(f"Combined hypergraph: {combined_hyper_index.shape[1]} connections")
    print()

    # Build heterogeneous graph
    print("Building heterogeneous graph...")
    # Get interaction edges from train_data (use original edge_index from Data object)
    interaction_edges = train_data.edge_index
    hetero_edge_index, hetero_edge_type = build_heterogeneous_graph(
        lncrna_knn_edges, drug_knn_edges, interaction_edges, num_lncrnas, num_drugs
    )
    print()

    # Prepare graph structures
    device = config['device']
    graphs = {
        'knn': knn_edge_index.to(device),
        'hyper': {
            'hyperedge_index': combined_hyper_index.to(device),
            'D_v': combined_D_v.to(device),
            'D_e': combined_D_e.to(device)
        },
        'hetero': {
            'edge_index': hetero_edge_index.to(device),
            'edge_type': hetero_edge_type.to(device)
        }
    }

    # Initialize model
    print("Initializing model...")
    model = TriFusion(
        in_channels=train_data.x.shape[1],  # 396 dimensions
        hidden_channels=config['hidden_channels'],
        out_channels=config['out_channels'],
        dropout=config['dropout']
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # Training loop
    print("Starting training...")
    print("=" * 80)
    best_test_auc = 0
    best_epoch = 0

    for epoch in range(1, config['epochs'] + 1):
        # Train
        loss = train_epoch(model, train_data, optimizer, graphs, device)

        # Evaluate every 10 epochs
        if epoch % 10 == 0:
            test_metrics = evaluate(model, test_data, graphs, device)
            test_auc, test_aupr, test_acc, test_f1, test_precision, test_recall = test_metrics

            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
                  f"Test AUC: {test_auc:.4f} | AUPR: {test_aupr:.4f} | "
                  f"ACC: {test_acc:.4f} | F1: {test_f1:.4f}")

            # Save best model
            if test_auc > best_test_auc:
                best_test_auc = test_auc
                best_epoch = epoch
                torch.save(model.state_dict(), f"{config['data_path']}/best_model.pth")

    print("=" * 80)
    print(f"Training completed! Best test AUC: {best_test_auc:.4f} at epoch {best_epoch}")
    print()

    # Load best model and final evaluation
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(f"{config['data_path']}/best_model.pth"))
    test_metrics = evaluate(model, test_data, graphs, device)
    test_auc, test_aupr, test_acc, test_f1, test_precision, test_recall = test_metrics

    result_df = pd.DataFrame({
        'metric': ['AUC', 'AUPR', 'Accuracy', 'F1', 'Precision', 'Recall'],
        'value': [test_auc, test_aupr, test_acc, test_f1, test_precision, test_recall]
    })
    result_df.to_csv(f'result/{config['data_path']}_test_results.csv', index=False)
    # Print final results
    print()
    print("=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)
    print(f"AUC:       {test_auc:.4f}")
    print(f"AUPR:      {test_aupr:.4f}")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"F1 Score:  {test_f1:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
