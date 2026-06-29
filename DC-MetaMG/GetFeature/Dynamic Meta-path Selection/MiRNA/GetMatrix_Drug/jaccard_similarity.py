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
with open('drug.txt', 'r') as infile:
    for line in infile:
        line = line.strip().split('\t')
        miRNA_names.append(line[1])


with open('Disease_drug.txt', 'r') as infile:
    for line in infile:
        line = line.strip().split('\t')
        disease_names.append(line[1])

#file_path = 'array_MD.txt'
file_path = 'array_DD.txt'

# 读取文件内容并将其存储到数组中
association_matrix = []
with open(file_path, 'r') as file:
    for line in file:
        # 去掉每行的首尾空白字符，并将每行的内容按空格分割成列表
        row = line.strip().split()
        # 将字符串列表转换为整数列表，并添加到数组中
        association_matrix.append(list(map(int, row)))
association_matrix = np.array(association_matrix)

# 转换为DataFrame便于查看
assoc_df = pd.DataFrame(association_matrix, index=miRNA_names, columns=disease_names)

def calculate_jaccard_similarity(matrix):
    n = matrix.shape[0]  # miRNA数量
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Jaccard相似度 = 交集大小 / 并集大小
            intersection = np.sum(np.logical_and(matrix[i], matrix[j]))
            union = np.sum(np.logical_or(matrix[i], matrix[j]))
            similarity_matrix[i, j] = intersection / union if union != 0 else 0

    return similarity_matrix


jaccard_sim = calculate_jaccard_similarity(association_matrix)

with open('Feature Matrix/Jaccard_sim_Drug.txt', 'w') as file:
    for row in jaccard_sim:
        file.write(' '.join(map(str, row)) + '\n')

jaccard_df = pd.DataFrame(jaccard_sim, index=miRNA_names, columns=miRNA_names)
print("\nJaccard相似性矩阵:")
print(jaccard_df)

import matplotlib.pyplot as plt
import seaborn as sns

def plot_similarity_matrix(sim_matrix, title, miRNA_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(sim_matrix, annot=True, cmap='YlOrRd',
                xticklabels=miRNA_names, yticklabels=miRNA_names)
    plt.title(title)
    plt.show()

#plot_similarity_matrix(jaccard_sim, "miRNA功能相似性 (Jaccard)", miRNA_names)
#对于稀疏二元数据(0/1)，Jaccard通常表现较好
