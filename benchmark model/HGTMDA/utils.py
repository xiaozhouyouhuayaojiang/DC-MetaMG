import torch
import random
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.data import Data
import pandas as pd

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def calculate_metrics(y_true, y_pred):
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-10)
    sensitivity = TP / (TP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    specificity = TN / (TN + FP + 1e-10)
    mcc = (TP*TN-FP*FN)/np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))
    F1_score = 2*(precision*sensitivity)/(precision+sensitivity + 1e-10)

    return accuracy, sensitivity, precision, specificity, F1_score, mcc


def get_data(data_ID, output_dim):
    if data_ID == 'miRNA':
        miRNA = np.loadtxt('data/miRNA/Gaussian_sim_MiRNA.txt')  # (541, 541)
        SM = np.loadtxt('data/miRNA/GIP_Topo_drug.txt')  # (831, 831)
        miRNA_sq = np.loadtxt('data/miRNA/Feature_MiRNA_SQ.txt')  # (541, 541)
        SM_mg = np.loadtxt('data/miRNA/drug_feature.txt')  # (831, 831)

        drug_ncRNA = pd.read_csv('data/miRNA/edges.txt', header=None, sep='\t')

    else:
        miRNA = np.loadtxt('data/lncRNA/Gaussian_sim_LncRNA.txt')  # (541, 541)
        SM = np.loadtxt('data/lncRNA/GIP_Topo_drug.txt')  # (831, 831)
        miRNA_sq = np.loadtxt('data/lncRNA/Feature_LncRNA_Sq.txt')  # (541, 541)
        SM_mg = np.loadtxt('data/lncRNA/drug_feature.txt')  # (831, 831)

        drug_ncRNA = pd.read_csv('data/lncRNA/edges.txt', header=None, sep='\t')

    drug_list = list(range(len(SM)))
    mirna_list = list(range(len(miRNA)))
    adj = torch.LongTensor(
        [[drug_list.index(int(x[0])), mirna_list.index(int(x[1])) + len(drug_list)]
         for x in drug_ncRNA.values]
    ).T

    m_emb = []
    m_emb_sq = []
    for m in range(len(miRNA)):
        m_emb.append(miRNA[m].tolist())
        m_emb_sq.append(miRNA_sq[m].tolist())

    m_emb = torch.Tensor(m_emb)
    m_emb_sq = torch.Tensor(m_emb_sq)

    s_emb = []
    s_emb_mg = []
    for s in range(len(SM)):
        s_emb.append(SM[s].tolist())
        s_emb_mg.append(SM_mg[s].tolist())

    s_emb = torch.Tensor(s_emb)
    s_emb_mg = torch.Tensor(s_emb_mg)

    feature = torch.cat([s_emb, m_emb])
    feature1 = torch.cat([s_emb_mg, m_emb_sq])
    feature_all = torch.stack([feature, feature1],dim=0).cuda()



    data = Data(x=feature, edge_index=adj).cuda()

    train_data, _, test_data = T.RandomLinkSplit(num_val=0, num_test=0.2,
                                                 is_undirected=True, split_labels=True,
                                                 add_negative_train_samples=True)(data)


    train_data.X = test_data.X = feature_all

    splits = dict(train=train_data, test=test_data)
    return splits


if __name__ == '__main__':
    data = get_data(2, 1024)
