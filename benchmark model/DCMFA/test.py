import warnings
import random
import numpy as np
import torch
import tqdm
import pandas as pd
import os

from sklearn.metrics import (roc_auc_score, f1_score,
                             average_precision_score)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from util import (load_dual_channel_data, split_data_dual,
                  split_train_test_dual, calculate_metrics)
from model import DCMFA

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
seed = 2026
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Channel assignments:
#   CH1 (FAT) – miRNA k-mer sequence  ↔  drug bond-angle  → 'mirna' folder
#   CH2 (FAV) – miRNA func-similarity ↔  drug bio features → 'lncRNA' folder
TYPE_CH1 = 'miRNA'


PE_K        = 16     # Laplacian PE dimensions
FAT_D_MODEL = 396
FAT_HEADS   = 12
FAT_LAYERS  = 2
FAV_LATENT  = 64
FAV_HIDDEN  = 128
MLP_HID1    = 256
MLP_HID2    = 64
DROPOUT     = 0.5
WEIGHT      = 0.5
EPOCHS      = 2000
LR          = 5e-3
WEIGHT_DECAY= 5e-4
VAE_BETA    = 0.5   # weight for KL term in VAE loss
N_FOLDS     = 1

os.makedirs('result', exist_ok=True)

# ─────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────
bce_loss = torch.nn.BCELoss()

def vae_loss(recon, x_orig, mu, logvar, beta=VAE_BETA):
    recon_loss = torch.nn.functional.mse_loss(recon, x_orig)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl


def total_loss(pred, label, recon, x_orig, mu, logvar):
    return bce_loss(pred, label) + vae_loss(recon, x_orig, mu, logvar)


# ─────────────────────────────────────────────
# Load & balance data
# ─────────────────────────────────────────────
print("Loading data …")
pos_ch1, neg_ch1, pos_ch2, neg_ch2, adj_mirna, adj_drug = \
    load_dual_channel_data(TYPE_CH1)

pos_ch1, neg_ch1, pos_ch2, neg_ch2 = \
    split_data_dual(pos_ch1, neg_ch1, pos_ch2, neg_ch2)

print(f"Positive pairs: {len(pos_ch1)} | Negative pairs: {len(neg_ch1)}")

# Infer feature dims from first sample
dim_drug_angle  = len(pos_ch1[0][0])   # CH1
dim_mirna_seq = len(pos_ch1[0][1])   # CH1
dim_drug_bio  = len(pos_ch2[0][0])   # CH2
dim_mirna_sim   = len(pos_ch2[0][1])   # CH2
print(f"Dims – miRNA-seq:{dim_mirna_seq}  drug-angle:{dim_drug_angle}  "
      f"miRNA-sim:{dim_mirna_sim}  drug-bio:{dim_drug_bio}")

# ─────────────────────────────────────────────
# 5-Fold cross-validation
# ─────────────────────────────────────────────
colors = list(mcolors.TABLEAU_COLORS.keys())

all_auc, all_aupr, all_acc, all_f1, all_pre = [], [], [], [], []

for fold in range(N_FOLDS):
    print(f"\n{'='*50}")
    print(f"Fold {fold+1}/{N_FOLDS}")

    (tr_d1, tr_m1, tr_d2, tr_m2, tr_label,
     te_d1, te_m1, te_d2, te_m2, te_label) = \
        split_train_test_dual(pos_ch1, neg_ch1, pos_ch2, neg_ch2,
                              adj_drug, adj_mirna, fold, pe_k=PE_K)

    # Move to device
    tr_d1, tr_m1 = (tr_d1.to(device), tr_m1.to(device))
    tr_d2, tr_m2, tr_label          = tr_d2.to(device), tr_m2.to(device), tr_label.to(device)
    te_d1, te_m1 = (te_d1.to(device), te_m1.to(device))
    te_d2, te_m2, te_label          = te_d2.to(device), te_m2.to(device), te_label.to(device)

    # Original input for VAE reconstruction target
    x_orig_train = torch.cat([tr_d2, tr_m2], dim=-1)

    model = DCMFA(
        dim_mirna_seq=dim_mirna_seq,
        dim_drug_angle=dim_drug_angle,
        dim_mirna_sim=dim_mirna_sim,
        dim_drug_bio=dim_drug_bio,
        fat_d_model=FAT_D_MODEL,
        fat_n_heads=FAT_HEADS,
        fat_n_layers=FAT_LAYERS,
        fat_pe_k=PE_K,
        fav_latent=FAV_LATENT,
        fav_hidden=FAV_HIDDEN,
        mlp_hid1=MLP_HID1,
        mlp_hid2=MLP_HID2,
        dropout=DROPOUT,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_loss = float('inf')
    best_state = None

    for epoch in tqdm.tqdm(range(EPOCHS), desc=f'Fold {fold+1} Training'):
        model.train()
        pred, recon, mu, logvar = model(tr_d1, tr_m1, tr_d2, tr_m2)
        loss = total_loss(pred, tr_label.unsqueeze(-1), recon, x_orig_train, mu, logvar)

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        opt.zero_grad()
        loss.backward()
        opt.step()

    # Evaluate with best checkpoint
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_te, _, _, _ = model(te_d1, te_m1, te_d2, te_m2)
        pred_te = pred_te.cpu()

    te_label_cpu = te_label.cpu()
    aucc = roc_auc_score(te_label_cpu.unsqueeze(-1), pred_te)
    aupr = average_precision_score(te_label_cpu.numpy(), pred_te.numpy())

    temp = pred_te.clone()
    temp[temp >= 0.5] = 1
    temp[temp  < 0.5] = 0
    acc, sen, pre, spe = calculate_metrics(te_label_cpu, temp)
    f1 = f1_score(te_label_cpu, temp.cpu())

    print(f"AUC:{aucc*100:.2f}  AUPR:{aupr*100:.2f}  "
          f"Acc:{acc*100:.2f}  F1:{f1*100:.2f}  Pre:{pre*100:.2f}")

    all_auc.append(aucc);  all_aupr.append(aupr)
    all_acc.append(acc);   all_f1.append(f1);  all_pre.append(pre)

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
print(f"\n{'='*50}")
print("5-Fold Cross-Validation Results")
print(f"AUC  : {np.mean(all_auc)*100:.2f} ± {np.std(all_auc)*100:.2f}")
print(f"AUPR : {np.mean(all_aupr)*100:.2f} ± {np.std(all_aupr)*100:.2f}")
print(f"Acc  : {np.mean(all_acc)*100:.2f} ± {np.std(all_acc)*100:.2f}")
print(f"F1   : {np.mean(all_f1)*100:.2f} ± {np.std(all_f1)*100:.2f}")
print(f"Pre  : {np.mean(all_pre)*100:.2f} ± {np.std(all_pre)*100:.2f}")

result_df = pd.DataFrame({
    'fold':      list(range(1, N_FOLDS+1)),
    'AUC':       all_auc,
    'AUPR':      all_aupr,
    'Accuracy':  all_acc,
    'F1':        all_f1,
    'Precision': all_pre,
})
result_df.to_csv(f'result/{TYPE_CH1}_results.csv', index=False)
print("Results saved to result/dcmfa_cv_results.csv")
plt.show()
