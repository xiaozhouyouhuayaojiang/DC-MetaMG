import numpy as np
import torch
import dgl
from model.hgnn import HGNN
from rdkit import Chem
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, WeightAndSum, SumPooling
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from rdkit.Chem import AllChem
import torch.nn as nn
from util.util import *

class DrugMG(nn.Module):
    def __init__(self,
                 smiles,
                 num_hidden,
                 num_layer,
                 dropout=0.1
                 ):
        super(DrugMG, self).__init__()

        self.smiles = smiles
        self.device = torch.device('cuda:0')
        self.drug_num = len(smiles)

        self.mol = Mol(smiles, num_hidden, num_layer, self.device)
        self.mol_size = self.mol.gnn.get_output_size()
        self.mol_fc = nn.Sequential(nn.Linear(self.mol_size, self.mol_size),
                                        nn.BatchNorm1d(self.mol_size),
                                        nn.Dropout(dropout),
                                        nn.ReLU(),

                                        nn.Linear(self.mol_size, self.mol_size),
                                        nn.BatchNorm1d(self.mol_size),
                                        nn.Dropout(dropout),
                                        nn.ReLU(),

                                        nn.Linear(self.mol_size, self.mol_size),
                                        nn.BatchNorm1d(self.mol_size),
                                        nn.Dropout(dropout),
                                        nn.ReLU()
                                        )
    def getMGemb(self):
        mol_emb = self.mol()
        mol_emb = self.mol_fc(mol_emb)
        drugMG_array = mol_emb.detach().cpu().numpy()
        return drugMG_array

class Mol(nn.Module):
    def __init__(self, smiles, num_hidden, num_layer, device='cuda:0'):
        super(Mol, self).__init__()
        self.device = device
        self.smiles = smiles
        self.readout = AvgPooling()

        mol_g = graph_construction(smiles)
        self.mol_g = dgl.batch(mol_g).to(self.device)

        nodes_type = self.mol_g.ndata['atomic_number'].tolist()
        nodes = []
        for i in range(len(nodes_type)):
                nodes.append([i, nodes_type[i]])

        nodes = torch.tensor(nodes).to(self.device)
        self.gnn = HGNN(self.mol_g, self.mol_g.edata['bond_type'], nodes, num_hidden, num_layer).to(device)

    def forward(self):
        a = self.gnn()
        result = self.readout(self.mol_g, a)
        return result

class CustomBondFeaturizer(PretrainBondFeaturizer):

    def __init__(self, bond_types=None, bond_direction_types=None, self_loop=True):
        if bond_types is None:
            bond_types = [
                Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC,
                Chem.rdchem.BondType.DATIVE  # 添加 DATIVE 键类型
            ]
        super(CustomBondFeaturizer, self).__init__(bond_types, bond_direction_types, self_loop)

    def __call__(self, mol):

        edge_features = []
        num_bonds = mol.GetNumBonds()
        if num_bonds == 0:
            assert self._self_loop, \
                'The molecule has 0 bonds and we should set self._self_loop to True.'

        # Compute features for each bond
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            bond_type = bond.GetBondType()
            bond_dir = bond.GetBondDir()
            if bond_type in self._bond_types:
                bond_feats = [
                    self._bond_types.index(bond_type),
                    self._bond_direction_types.index(bond_dir)
                ]
            else:
                raise ValueError(f"Unknown bond type: {bond_type}")
            edge_features.extend([bond_feats, bond_feats.copy()])

        if self._self_loop:
            self_loop_features = torch.zeros((mol.GetNumAtoms(), 2), dtype=torch.int64)
            self_loop_features[:, 0] = len(self._bond_types)

        if num_bonds == 0:
            edge_features = self_loop_features
        else:
            edge_features = np.stack(edge_features)
            edge_features = torch.from_numpy(edge_features.astype(np.int64))
            if self._self_loop:
                edge_features = torch.cat([edge_features, self_loop_features], dim=0)

        return {'bond_type': edge_features[:, 0], 'bond_direction_type': edge_features[:, 1]}

def graph_construction(smiles):
    graphs = []

    for smi in smiles:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        g = mol_to_bigraph(mol, add_self_loop=True,
                           node_featurizer=PretrainAtomFeaturizer(),
                           edge_featurizer=CustomBondFeaturizer(),
                           canonical_atom_order=False)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)

        coords = mol.GetConformer().GetPositions()
        geom_features = []
        for i in range(mol.GetNumAtoms()):
            neighbors = [x.GetIdx() for x in mol.GetAtomWithIdx(i).GetNeighbors()]
            if len(neighbors) >= 2:
                # 计算距离
                dists = cdist([coords[i]], coords[neighbors]).flatten()
                # 计算角度
                angles = []
                for j in range(len(neighbors)):
                    for k in range(j + 1, len(neighbors)):
                        v1 = coords[neighbors[j]] - coords[i]
                        v2 = coords[neighbors[k]] - coords[i]
                        angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
                        angles.append(angle)
                # 组合特征
                geom_features.append([
                    np.mean(dists), np.std(dists),  # 距离统计
                    np.mean(angles), np.std(angles),  # 角度统计
                    len(neighbors)  # 配位数
                ])
            else:
                geom_features.append([0] * 5)  # 默认值

        # 添加到节点特征
        geom_features = np.array(geom_features)

        # 归一化处理
        scaler = StandardScaler()
        geom_features = scaler.fit_transform(geom_features)
        geom_features = torch.tensor(geom_features, dtype=torch.float32)
        g.ndata.update({'3D': geom_features})

        graphs.append(g)

    return graphs


