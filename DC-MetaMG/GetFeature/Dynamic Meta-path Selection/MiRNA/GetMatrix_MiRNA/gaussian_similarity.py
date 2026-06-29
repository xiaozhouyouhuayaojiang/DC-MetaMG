import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from sklearn.metrics import jaccard_score

# 示例数据 - 实际应用中替换为你的数据
miRNA_names = []
disease_names = []
'''
with open('miRNA.txt','r') as infile:
    for line in infile:
        line = line.strip().split('\t')
        miRNA_names.append(line[1])
'''
with open('miRNA.txt', 'r') as infile:
    for line in infile:
        line = line.strip().split('\t')
        miRNA_names.append(line[1])


with open('Disease.txt', 'r') as infile:
    for line in infile:
        line = line.strip().split('\t')
        disease_names.append(line[1])



#file_path = 'array_MD.txt'
file_path = 'MD_ARR.txt'


association_matrix = []
with open(file_path, 'r') as file:
    for line in file:

        row = line.strip().split()

        association_matrix.append(list(map(int, row)))


association_matrix = np.array(association_matrix)

assoc_df = pd.DataFrame(association_matrix, index=miRNA_names, columns=disease_names)


def calculate_gaussian_similarity(matrix, gamma=1.0):
    n = matrix.shape[0]
    similarity_matrix = np.zeros((n, n))


    norms = np.array([np.linalg.norm(row) for row in matrix])

    for i in range(n):
        for j in range(n):

            dist_sq = np.sum((matrix[i] - matrix[j]) ** 2)

            similarity_matrix[i, j] = np.exp(
                -gamma * dist_sq / (norms[i] * norms[j] if norms[i] * norms[j] != 0 else 1))

    return similarity_matrix

from sklearn.decomposition import PCA
gaussian_sim = calculate_gaussian_similarity(association_matrix)
pca = PCA(n_components=198)
emb = pca.fit_transform(gaussian_sim)

with open('source/Gaussian_sim_MiRNA.txt', 'w') as file:
    for row in emb:
        file.write(' '.join(map(str, row)) + '\n')

gaussian_df = pd.DataFrame(gaussian_sim, index=miRNA_names, columns=miRNA_names)

print(gaussian_df)


import matplotlib.pyplot as plt
import seaborn as sns

def plot_similarity_matrix(sim_matrix, title, miRNA_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(sim_matrix, annot=True, cmap='YlOrRd',
                xticklabels=miRNA_names, yticklabels=miRNA_names)
    plt.title(title)
    plt.show()
