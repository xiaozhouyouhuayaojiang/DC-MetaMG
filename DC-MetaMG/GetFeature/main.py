import argparse

from trainer import Trainer
from model.model import MetaMG
from util.util import get_data, set_seed


parser = argparse.ArgumentParser(description="DC-MetaMG Stage 1: Feature Extraction")

# Dataset
parser.add_argument('--dataset', type=str, default='LncRNA',
                    help='Dataset name: MiRNA or LncRNA')
parser.add_argument('--data_path', type=str, default='data/LncRNA',
                    help='Path to dataset directory')
parser.add_argument('--n_Drug', type=int, default=154,
                    help='Number of drug nodes (MiRNA: 60, LncRNA: 154)')
parser.add_argument('--n_NcRNA', type=int, default=955,
                    help='Number of ncRNA nodes (MiRNA: 561, LncRNA: 955)')
parser.add_argument('--embedding_size', type=int, default=396,
                    help='Embedding dimension (MiRNA: 198, LncRNA: 396)')

# Training
parser.add_argument('--seed', type=int, default=2026,
                    help='Random seed for reproducibility')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--train_batch_size', type=int, default=4096)
parser.add_argument('--eval_batch_size', type=int, default=4096000)

# Model
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--hyper_layers', type=int, default=1)
parser.add_argument('--num_clusters', type=int, default=3)
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--proto_reg', type=float, default=1e-8)
parser.add_argument('--reg_weight', type=float, default=1e-1)
parser.add_argument('--delta', type=int, default=5)
parser.add_argument('--m_step', type=int, default=1)
parser.add_argument('--warm_up_step', type=int, default=20)

# Gradient clipping
parser.add_argument('--clip_max_norm', type=float, default=30,
                    help='Maximum norm for gradient clipping')
parser.add_argument('--clip_norm_type', type=int, default=2,
                    help='Norm type for gradient clipping (2 = L2)')

args = parser.parse_args()

# Set seed before everything else
set_seed(args.seed)

# Load data and train/test splits (splits are persisted to data_path/splits.pt)
smiles, splits = get_data(args)

# Build model with training graph
model = MetaMG(args, smiles, args.embedding_size, 1, splits['train']).to(args.device)

# Train and export embeddings + edge split indices for DC_Prediction
trainer = Trainer(args, model, splits)
trainer.train(splits['train'])
