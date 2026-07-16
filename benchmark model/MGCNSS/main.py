"""
Main training and evaluation script for MGCNSS model
Comparative experiment with proper train/val/test splitting and no data leakage
"""
import csv
import os
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from model import MHGCN, LinkPredictor
from utils import (
    set_seed, load_data, split_data,
    build_bipartite_adj, evaluate_model, print_metrics
)


def parse_args():
    parser = argparse.ArgumentParser(description='MGCNSS Model Training')
    parser.add_argument('--type', type=str,
                        default='miRNA',
                        help='Path to data directory')
    parser.add_argument('--data_dir', type=str,
                        default='miRNA\\',
                        help='Path to data directory')
    parser.add_argument('--save_dir', type=str,
                        default='result\\miRNA_metrics',
                        help='Path to data directory')
    parser.add_argument('--seed', type=int, default=2026,
                        help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension (unused, kept for compat)')
    parser.add_argument('--output_dim', type=int, default=198,
                        help='Output embedding dimension')
    parser.add_argument('--proj_dim', type=int, default=198,
                        help='Internal feature projection dimension inside MHGCN')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='Number of GCN layers (1-5)')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--num_runs', type=int, default=5,
                        help='Number of runs with different seeds')
    parser.add_argument('--margin', type=float, default=0.5,
                        help='Margin for ranking loss auxiliary term')
    parser.add_argument('--margin_weight', type=float, default=0.1,
                        help='Weight of margin ranking loss')

    return parser.parse_args()


def train_epoch(model, predictor, drug_raw, ncrna_raw, adj_bipartite,
                train_pos, train_neg, optimizer, num_drugs, num_ncrnas, device,
                margin=0.5, margin_weight=0.1):
    model.train()
    predictor.train()
    optimizer.zero_grad()

    embeddings = model(drug_raw, ncrna_raw, adj_bipartite, num_drugs, num_ncrnas)

    pos_src_emb = embeddings[train_pos[:, 0]]
    pos_dst_emb = embeddings[train_pos[:, 1]]
    neg_src_emb = embeddings[train_neg[:, 0]]
    neg_dst_emb = embeddings[train_neg[:, 1]]

    pos_scores = predictor(pos_src_emb, pos_dst_emb)
    neg_scores = predictor(neg_src_emb, neg_dst_emb)

    pos_loss = -torch.mean(F.logsigmoid(pos_scores))
    neg_loss = -torch.mean(F.logsigmoid(-neg_scores))
    bce_loss = pos_loss + neg_loss

    n = min(len(pos_scores), len(neg_scores))
    rank_loss = F.margin_ranking_loss(
        pos_scores[:n], neg_scores[:n],
        target=torch.ones(n, device=device),
        margin=margin
    )
    loss = bce_loss + margin_weight * rank_loss

    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate_epoch(model, predictor, drug_raw, ncrna_raw, adj_bipartite,
                   eval_pos, eval_neg, num_drugs, num_ncrnas, device):
    model.eval()
    predictor.eval()

    with torch.no_grad():
        embeddings = model(drug_raw, ncrna_raw, adj_bipartite, num_drugs, num_ncrnas)

        pos_edges_adjusted = eval_pos.copy()


        neg_edges_adjusted = eval_neg.copy()


        metrics = evaluate_model(
            model, predictor,
            embeddings.cpu(),
            pos_edges_adjusted, neg_edges_adjusted,
            device
        )

    return metrics


def train_and_evaluate(args, run_seed):
    set_seed(run_seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    print("Loading data...")
    drug_features, ncrna_features, interactions = load_data(args.data_dir, args.type)

    num_drugs = len(drug_features)
    num_ncrnas = len(ncrna_features)
    print(f"Number of drugs: {num_drugs}")
    print(f"Number of ncRNAs: {num_ncrnas}")
    print(f"Number of interactions: {len(interactions)}")

    print("\nSplitting data...")
    train_pos, val_pos, test_pos, train_neg, val_neg, test_neg = split_data(
        interactions, num_drugs, num_ncrnas,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=run_seed
    )

    print(f"Train: {len(train_pos)} positive, {len(train_neg)} negative")
    print(f"Val:   {len(val_pos)} positive, {len(val_neg)} negative")
    print(f"Test:  {len(test_pos)} positive, {len(test_neg)} negative")

    train_set = set(map(tuple, train_pos))
    val_set = set(map(tuple, val_pos))
    test_set = set(map(tuple, test_pos))
    assert len(train_set & val_set) == 0, "Train-Val overlap detected!"
    assert len(train_set & test_set) == 0, "Train-Test overlap detected!"
    assert len(val_set & test_set) == 0, "Val-Test overlap detected!"
    print("No data leakage: Train/Val/Test sets are disjoint")

    # Only bipartite interaction adjacency is used (no homogeneous similarity graphs)
    adj_bipartite = build_bipartite_adj(train_pos, num_drugs, num_ncrnas).to(device)

    # Keep raw features as separate tensors for type-specific projection inside MHGCN
    drug_raw = torch.FloatTensor(drug_features).to(device)
    ncrna_raw = torch.FloatTensor(ncrna_features).to(device)

    drug_in_dim = drug_features.shape[1]
    ncrna_in_dim = ncrna_features.shape[1]

    print(f"\nDrug feature dim: {drug_in_dim}, ncRNA feature dim: {ncrna_in_dim}")
    print(f"Internal projection dim: {args.proj_dim}, output dim: {args.output_dim}")

    model = MHGCN(
        drug_in_dim=drug_in_dim,
        ncrna_in_dim=ncrna_in_dim,
        nfeat=args.proj_dim,
        nhid=args.hidden_dim,
        out=args.output_dim,
        dropout=args.dropout,
        num_layers=args.num_layers
    ).to(device)

    predictor = LinkPredictor(in_dim=args.output_dim, dropout=args.dropout).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    total_params = (sum(p.numel() for p in model.parameters()) +
                    sum(p.numel() for p in predictor.parameters()))
    print(f"\nModel: {args.num_layers} layers, dropout={args.dropout}, params={total_params}")

    train_pos_t = torch.LongTensor(train_pos)
    train_neg_t = torch.LongTensor(train_neg)
    print(f"\n{'='*60}")
    print("Training...")
    print(f"{'='*60}")

    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    best_pred_state = None

    for epoch in range(args.epochs):
        loss = train_epoch(
            model, predictor, drug_raw, ncrna_raw, adj_bipartite,
            train_pos_t, train_neg_t,
            optimizer, num_drugs, num_ncrnas, device,
            margin=args.margin, margin_weight=args.margin_weight
        )

        val_metrics = evaluate_epoch(
            model, predictor, drug_raw, ncrna_raw, adj_bipartite,
            val_pos, val_neg,
            num_drugs, num_ncrnas, device
        )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Loss: {loss:.4f} | "
                  f"Val AUC: {val_metrics['auc']:.4f} | "
                  f"Val AUPR: {val_metrics['aupr']:.4f}")

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_pred_state = {k: v.clone() for k, v in predictor.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)
    predictor.load_state_dict(best_pred_state)
    print(f"\nBest validation AUC: {best_val_auc:.4f} at epoch {best_epoch+1}")

    print(f"\n{'='*60}")
    print("Final Test Evaluation")
    print(f"{'='*60}")

    test_metrics = evaluate_epoch(
        model, predictor, drug_raw, ncrna_raw, adj_bipartite,
        test_pos, test_neg,
        num_drugs, num_ncrnas, device
    )

    print_metrics("Test", test_metrics)

    return test_metrics


def main():
    args = parse_args()

    print("="*60)
    print("MGCNSS Model - Comparative Experiment")
    print("="*60)
    print(f"Configuration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Base seed: {args.seed}")
    print(f"  Number of runs: {args.num_runs}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Proj dim: {args.proj_dim}")
    print(f"  Early stopping patience: {args.patience}")

    all_results = {
        'auc': [],
        'aupr': [],
        'f1': [],
        'recall': [],
        'accuracy': []
    }

    seeds = [args.seed + i * 100 for i in range(args.num_runs)]

    for run_idx, seed in enumerate(seeds):
        print(f"\n{'#'*60}")
        print(f"# RUN {run_idx+1}/{args.num_runs} (seed={seed})")
        print(f"{'#'*60}")

        test_metrics = train_and_evaluate(args, seed)

        all_results['auc'].append(test_metrics['auc'])
        all_results['aupr'].append(test_metrics['aupr'])
        all_results['f1'].append(test_metrics['f1'])
        all_results['recall'].append(test_metrics['recall'])
        all_results['accuracy'].append(test_metrics['accuracy'])

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS (Mean ± Std over {args.num_runs} runs)")
    print(f"{'='*60}")
    print(f"AUC:      {np.mean(all_results['auc']):.4f} ± {np.std(all_results['auc']):.4f}")
    print(f"AUPR:     {np.mean(all_results['aupr']):.4f} ± {np.std(all_results['aupr']):.4f}")
    print(f"F1:       {np.mean(all_results['f1']):.4f} ± {np.std(all_results['f1']):.4f}")
    print(f"Recall:   {np.mean(all_results['recall']):.4f} ± {np.std(all_results['recall']):.4f}")
    print(f"Accuracy: {np.mean(all_results['accuracy']):.4f} ± {np.std(all_results['accuracy']):.4f}")

    print(f"\n{'='*60}")
    print("Experiment completed successfully!")
    print(f"{'='*60}")

    file_exists = os.path.isfile(args.save_dir)
    with open(args.save_dir, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['split', 'auc', 'aupr', 'f1', 'recall', 'accuracy'])

        writer.writerow([
            f"Test",
            f"{np.mean(all_results['auc']):.4f}",
            f"{np.mean(all_results['aupr']):.4f}",
            f"{np.mean(all_results['f1']):.4f}",
            f"{np.mean(all_results['recall']):.4f}",
            f"{np.mean(all_results['accuracy']):.4f}"
        ])


if __name__ == '__main__':
    main()

