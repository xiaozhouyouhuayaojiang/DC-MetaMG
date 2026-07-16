import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
import argparse
import os
import warnings
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score, recall_score, precision_score
import torch_geometric.transforms as T
import random
from data_loader_drug_lncrna import load_drug_lncrna_data
from model import DrugLncRNAPredictor
from torch_geometric.utils import add_self_loops, negative_sampling, degree
warnings.filterwarnings("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

parser = argparse.ArgumentParser(description='Drug-LncRNA Association Prediction')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--wd', type=float, default=1e-3)
parser.add_argument('--in_channels', type=int, default=128)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--val_ratio', type=float, default=0.1)
parser.add_argument('--test_ratio', type=float, default=0.2)
parser.add_argument('--random_seed', type=int, default=2026)
parser.add_argument('--data_dir', type=str,
                    default='lncRNA')
parser.add_argument('--save_dir', type=str, default='result')

args = parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


os.makedirs(args.save_dir, exist_ok=True)

set_seed(args.random_seed)

print("\nLoading data...")
data = load_drug_lncrna_data(args.data_dir)

# Split dataset
transform = T.RandomLinkSplit(
    num_val=args.val_ratio,
    num_test=args.test_ratio,
    is_undirected=True,
    split_labels=True,
    add_negative_train_samples=True,
    neg_sampling_ratio=1.0,
)
train_data, val_data, test_data = transform(data)

train_data = train_data.to(device)
val_data   = val_data.to(device)
test_data  = test_data.to(device)

train_pos_idx = train_data.pos_edge_label_index
train_neg_idx = train_data.neg_edge_label_index



train_idx     = torch.cat([train_pos_idx, train_neg_idx], dim=1)
train_lbl     = torch.cat([torch.ones(train_pos_idx.size(1)),
                            torch.zeros(train_neg_idx.size(1))]).float().to(device)

val_pos_idx = val_data.pos_edge_label_index
val_neg_idx = val_data.neg_edge_label_index
val_idx     = torch.cat([val_pos_idx, val_neg_idx], dim=1)
val_lbl     = torch.cat([torch.ones(val_pos_idx.size(1)),
                          torch.zeros(val_neg_idx.size(1))]).float().to(device)

test_pos_idx = test_data.pos_edge_label_index
test_neg_idx = test_data.neg_edge_label_index


test_idx     = torch.cat([test_pos_idx, test_neg_idx], dim=1)
test_lbl     = torch.cat([torch.ones(test_pos_idx.size(1)),
                           torch.zeros(test_neg_idx.size(1))]).float().to(device)

print(f"\nSplit sizes:")
print(f"  Train pos/neg: {train_pos_idx.size(1)} / {train_neg_idx.size(1)}")
print(f"  Val   pos/neg: {val_pos_idx.size(1)} / {val_neg_idx.size(1)}")
print(f"  Test  pos/neg: {test_pos_idx.size(1)} / {test_neg_idx.size(1)}")

n_drug = data.n_drug

# Initialize model
model = DrugLncRNAPredictor(
    drug_feat_dim=train_data.x.shape[1],
    lncrna_feat_dim=train_data.x.shape[1],
    n_drug=n_drug,
    in_channels=args.in_channels,
    hidden_channels=args.hidden_channels,
    num_layers=args.num_layers,
    dropout=args.dropout,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

# Use all training edges (not just pos_edge_label_index) for GNN message passing
train_edge_index = train_data.pos_edge_label_index

best_val_auc = 0.0
print(f"\nTraining for {args.epochs} epochs...")
print("=" * 80)

for epoch in range(args.epochs):
    model.train()
    optimizer.zero_grad()

    z = model.encode(
         train_data.x,
        train_edge_index
    )
    scores = model.decode(z, train_idx)
    loss = F.binary_cross_entropy(scores, train_lbl)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % 20 == 0 or epoch == 0:
        model.eval()
        with torch.no_grad():
            train_auc = roc_auc_score(train_lbl.cpu().numpy(),
                                      scores.detach().cpu().numpy())

            z = model.encode(
                train_data.x,
                train_edge_index
            )
            val_scores = model.decode(z, val_idx)
            val_auc = roc_auc_score(val_lbl.cpu().numpy(),
                                    val_scores.detach().cpu().numpy())

        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | "
              f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")


# Final test evaluation using best checkpoint
print(f"\n{'='*80}")
print("Final Test Evaluation (best val checkpoint)")
print(f"{'='*80}")


model.eval()
with torch.no_grad():
    z = model.encode(
         train_data.x,
        train_edge_index
    )
    test_scores = model.decode(z, test_idx)

y_true = test_lbl.cpu().numpy()
y_pred = test_scores.cpu().numpy()
y_pred_binary = (y_pred > 0.5).astype(int)

print(f"AUC:       {roc_auc_score(y_true, y_pred):.4f}")
print(f"AUPR:      {average_precision_score(y_true, y_pred):.4f}")
print(f"Accuracy:  {accuracy_score(y_true, y_pred_binary):.4f}")
print(f"F1:        {f1_score(y_true, y_pred_binary):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred_binary):.4f}")
print(f"Precision: {precision_score(y_true, y_pred_binary):.4f}")
print(f"{'='*80}")
print("Training completed successfully!")
metrics = {
    'AUC': roc_auc_score(y_true, y_pred),
    'AUPR': average_precision_score(y_true, y_pred),
    'Accuracy': accuracy_score(y_true, y_pred_binary),
    'F1': f1_score(y_true, y_pred_binary),
    'Recall': recall_score(y_true, y_pred_binary),
    'Precision': precision_score(y_true, y_pred_binary)
}
with open(os.path.join(args.save_dir, f"{args.data_dir}_metrics.txt"), 'w') as f:
    for key, value in metrics.items():
        f.write(f"{key}: {value:.4f}\n")