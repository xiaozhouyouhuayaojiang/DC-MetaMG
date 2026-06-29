import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from sklearn.metrics import jaccard_score

miRNA_names = []
disease_names = []
'''
with open('miRNA.txt','r') as infile:
    for line in infile:
        line = line.strip().split('\t')
        miRNA_names.append(line[1])
'''
with open('drug.txt','r') as infile:
    for line in infile:
        line = line.strip().split('\t')
        miRNA_names.append(line[1])


with open('Disease_drug.txt','r') as infile:
    for line in infile:
        line = line.strip().split('\t')
        disease_names.append(line[1])

#file_path = 'array_MD.txt'
file_path = 'array_DD.txt'

association_matrix = []
with open(file_path, 'r') as file:
    for line in file:

        row = line.strip().split()

        association_matrix.append(list(map(int, row)))
association_matrix = np.array(association_matrix)


assoc_df = pd.DataFrame(association_matrix, index=miRNA_names, columns=disease_names)

def calculate_jaccard_similarity(matrix):
    n = matrix.shape[0]
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            intersection = np.sum(np.logical_and(matrix[i], matrix[j]))
            union = np.sum(np.logical_or(matrix[i], matrix[j]))
            similarity_matrix[i, j] = intersection / union if union != 0 else 0

    return similarity_matrix


jaccard_sim = calculate_jaccard_similarity(association_matrix)

with open('Feature Matrix/Jaccard_sim_Drug.txt', 'w') as file:
    for row in jaccard_sim:
        file.write(' '.join(map(str, row)) + '\n')

jaccard_df = pd.DataFrame(jaccard_sim, index=miRNA_names, columns=miRNA_names)

print(jaccard_df)

import matplotlib.pyplot as plt
import seaborn as sns

def plot_similarity_matrix(sim_matrix, title, miRNA_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(sim_matrix, annot=True, cmap='YlOrRd',
                xticklabels=miRNA_names, yticklabels=miRNA_names)
    plt.title(title)
    plt.show()

