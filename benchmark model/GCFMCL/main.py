import argparse
from recbole.config import Config
from util import *
from recbole.utils import init_seed
from trainer import GCFMCLTrainer
from model import GCFMCL
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='miRNA', help='datasets')


parser.add_argument('--n_Drug', type=int, default=60,
                    help='Number of drug nodes (MiRNA: 60, LncRNA: 154)')
parser.add_argument('--n_NcRNA', type=int, default=561,
                    help='Number of ncRNA nodes (MiRNA: 561, LncRNA: 955)')
parser.add_argument('--embedding_size', type=int, default=396,
                    help='Embedding dimension (MiRNA: 198, LncRNA: 396)')
# Training
parser.add_argument('--seed', type=int, default=2026,
                    help='Random seed for reproducibility')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=100)

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
parser.add_argument('--ssl_temp', type=int, default=1)
parser.add_argument('--ssl_reg', type=int, default=1)
parser.add_argument('--clip_max_norm', type=float, default=30,
                    help='Maximum norm for gradient clipping')
parser.add_argument('--clip_norm_type', type=int, default=2,
                    help='Norm type for gradient clipping (2 = L2)')

args, _ = parser.parse_known_args()

train_data, test_data = data_preparation(args.data_path)
model = GCFMCL(args, train_data).to(args.device)
trainer = GCFMCLTrainer(args, model)
trainer.fit(train_data)