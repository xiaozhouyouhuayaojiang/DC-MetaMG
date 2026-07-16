from sklearn.ensemble import RandomForestClassifier
from util import *
from model import mlp
import numpy as np
import torch
import warnings
import random
import tqdm

warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, roc_curve, auc, precision_recall_curve, \
    average_precision_score, f1_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d

colors = list(mcolors.TABLEAU_COLORS.keys())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

type = 'lncRNA'

seed = 2026
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def test():
    model.eval()
    with torch.no_grad():

        out = model(test_data.to(device))

        out_cpu = out.cpu()

    aucc = roc_auc_score(test_label.unsqueeze(-1).cpu(), out_cpu)
    temp = torch.tensor(out_cpu)
    temp[temp >= 0.5] = 1
    temp[temp < 0.5] = 0
    acc, sen, pre, spe = calculate_metrics(test_label.cpu(), temp)
    F1 = f1_score(test_label.cpu(), temp.cpu())
    aupr = average_precision_score(test_label.cpu().numpy(), out_cpu.numpy())
    print("auc:{},acc:{},pre:{},sen:{},f1:{},aupr:{}".format(aucc * 100, acc * 100, pre * 100, sen * 100, F1 * 100,
                                                             aupr * 100))
    result_df = pd.DataFrame({
        'metric': ['AUC', 'AUPR', 'Accuracy', 'F1', 'Precision'],
        'value': [aucc, aupr, acc, F1, pre]
    })
    result_df.to_csv(f'../result/{type}_test_results.csv', index=False)



pos_data, neg_data = load_data(type)
pos_list, neg_list = split_data(pos_data, neg_data)

pos_data = deal_embedding(pos_list)
neg_data = deal_embedding(neg_list)

train_data, train_label, test_data, test_label = split_train_test(pos_data, neg_data, 1)

train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
train_label = torch.tensor(train_label, dtype=torch.float32).to(device)
test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
test_label = torch.tensor(test_label, dtype=torch.float32).to(device)

model = mlp(len(train_data[0]), 256, 64, 1)

model = model.to(device)

opt = torch.optim.Adam(params=model.parameters(), lr=0.005, weight_decay=5e-4)
loss_fn = torch.nn.BCELoss()
best_loss = 1
best_model = model

for epoch in tqdm.tqdm(range(2000), desc=f'Training~'):
        model.train()
        out = model(train_data)
        loss = loss_fn(out, train_label.unsqueeze(-1))
        if loss < best_loss:
            best_model = model
        opt.zero_grad()
        loss.backward()
        opt.step()

test()

plt.show()