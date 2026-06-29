import numpy as np
import torch

drugs = set()
ncrnas = set()
edges = []
splits = torch.load("splits.pt")
data = splits['train'].pos_edge_label_index
association_matrix = np.zeros((154, 955), dtype=int)
for d, r in data.T:
    association_matrix[int(d), int(r)-154] = 1


association_matrix = np.array(association_matrix, dtype=int)

correlation_matrix = np.dot(association_matrix.T, association_matrix)

np.savetxt('../data/correlation_LncRNA.txt', correlation_matrix, fmt='%.2f')

correlation_matrix = np.dot(association_matrix, association_matrix.T)

np.savetxt('../data/correlation_Drug.txt', correlation_matrix, fmt='%.2f')

