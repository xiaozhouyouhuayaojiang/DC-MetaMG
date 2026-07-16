import torch
import argparse
from utils import get_data, MaskPath, print_result, set_seed
from model import GNNEncoder, EdgeDecoder, GAM
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--layer', default="gcn", help="sage, gcn, gin, gat, gat2")
parser.add_argument('--seed', type=int, default=2026, help="Random seed for model and dataset.")

parser.add_argument('--num_encoder', type=int, default=2, help="numbers of GNN encoder")
parser.add_argument('--num_decoder', type=int, default=2, help="numbers of Edge decoder")

parser.add_argument('--walk_length', type=int, default=3, help="length of walk")
parser.add_argument('--p', type=float, default=0.3, help='Mask ratio')

parser.add_argument('--lr', type=float, default=1e-3, help='learning rate in optimizer')
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay in optimizer')

parser.add_argument('--times', type=int, default=10, help="numbers of training times")
parser.add_argument('--epoch', type=int, default=30, help="numbers of training epoch")

args = parser.parse_args()

set_seed(args.seed)

type = "miRNA"
for i in range(1):
    data = get_data(type)  # utils.py: returns dict with 'train' and 'test' Data objects


    in_channels = data['train'].x.size(1)  # 396
    num_nodes   = data['train'].num_nodes  # 621

    mask = MaskPath(p=args.p, num_nodes=num_nodes, walk_length=args.walk_length)
    encoder = GNNEncoder(in_channels, 128, 256, num_layers=args.num_encoder, layer=args.layer)
    edge_decoder = EdgeDecoder(256, 64, 1, num_layers=args.num_decoder)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAM(encoder, edge_decoder, mask).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    all_result = []
    for x in range(args.times):
        for epoch in range(args.epoch):

            model.train()
            loss = model.train_epoch(data, optimizer)

        model.eval()
        test_data = data['test']
        z = model.encoder(test_data.x, test_data.edge_index)
        result = model.test(z, test_data.pos_edge_label_index, test_data.neg_edge_label_index, args.layer)
        all_result.append(result)

    df = print_result(all_result)
    df = pd.DataFrame(df)
    df.to_csv(f'result/{type}_results.csv', index=False)
    print("Results saved to result/dcmfa_cv_results.csv")
# plt.show()
