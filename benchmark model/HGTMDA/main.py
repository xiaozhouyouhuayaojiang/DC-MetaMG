import torch
import argparse
from mask import Mask
from utils import get_data, set_seed
from model import GNNEncoder, EdgeDecoder, DegreeDecoder, GMAE
import numpy as np
# main parameter
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="lncRNA", help="Choose Datasets (miRNA or lncRNA)")
parser.add_argument('--seed', type=int, default=2026, help="Random seed for model and dataset.")
parser.add_argument('--dim', type=int, default=396, help='Feature Dimension of Similarity Matrix(dataset1 >= 831, dataset2 >= 286)')

parser.add_argument('--alpha', type=float, default=0.007, help='loss weight for degree prediction.')
parser.add_argument('--p', type=float, default=0.6, help='Mask ratio')
args = parser.parse_args()
set_seed(args.seed)
auc_mean = []
for i in range(5):
    splits = get_data(args.dataset, args.dim)
    encoder = GNNEncoder(in_channels=args.dim, hidden_channels=64, out_channels=128)
    edge_decoder = EdgeDecoder(in_channels=128, hidden_channels=64, out_channels=1)
    degree_decoder = DegreeDecoder(in_channels=128, hidden_channels=64, out_channels=1)
    mask = Mask(p=args.p)

    model = GMAE(args, edge_decoder, degree_decoder, mask).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
    for epoch in range(1000):
        model.train()
        loss = model.train_epoch(splits['train'], optimizer, alpha=args.alpha)
    model.eval()
    test_data = splits['test']
    train_data = splits['train']

    z = model.encoder_Coa(train_data.x, train_data.edge_index)
    z1 = model.encoder_fine(train_data.x, train_data.edge_index)

    test_auc, test_ap, acc, sen, pre, spe, F1, mcc = model.test(z,z1, test_data.pos_edge_label_index, test_data.neg_edge_label_index)
    auc_mean.append(test_auc)
    results = {'AUC': "{:.6f}".format(test_auc),
           'AP': "{:.6f}".format(test_ap),
           "ACC": "{:.6f}".format(acc),
           "SEN": "{:.6f}".format(sen),
           "PRE": "{:.6f}".format(pre),
           "SPE": "{:.6f}".format(spe),
           "F1": "{:.6f}".format(F1),
           "MCC": "{:.6f}".format(mcc)
           }

    print(results)

print(np.mean(auc_mean))
with open(f'result/{args.dataset}_metrics.txt', 'w') as f:
        f.write(f"{np.mean(auc_mean)}")