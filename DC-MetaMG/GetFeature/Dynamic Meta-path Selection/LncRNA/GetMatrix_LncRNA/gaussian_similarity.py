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
with open('LncRNA_ID.txt', 'r') as infile:
    for line in infile:
        line = line.strip().split('\t')
        miRNA_names.append(line[1])


with open('Disease_ID.txt', 'r') as infile:
    for line in infile:
        line = line.strip().split('\t')
        disease_names.append(line[1])



#file_path = 'array_MD.txt'
file_path = 'LncD_ARR.txt'

# 读取文件内容并将其存储到数组中
association_matrix = []
with open(file_path, 'r') as file:
    for line in file:
        # 去掉每行的首尾空白字符，并将每行的内容按空格分割成列表
        row = line.strip().split()
        # 将字符串列表转换为整数列表，并添加到数组中
        association_matrix.append(list(map(int, row)))

# 将数组转换为 NumPy 数组
association_matrix = np.array(association_matrix)

# 转换为DataFrame便于查看
assoc_df = pd.DataFrame(association_matrix, index=miRNA_names, columns=disease_names)


def calculate_gaussian_similarity(matrix, gamma=1.0):
    n = matrix.shape[0]
    similarity_matrix = np.zeros((n, n))

    # 首先计算每个miRNA的相互作用谱范数
    norms = np.array([np.linalg.norm(row) for row in matrix])

    for i in range(n):
        for j in range(n):
            # 计算欧氏距离的平方
            dist_sq = np.sum((matrix[i] - matrix[j]) ** 2)
            # 计算高斯相似度
            similarity_matrix[i, j] = np.exp(
                -gamma * dist_sq / (norms[i] * norms[j] if norms[i] * norms[j] != 0 else 1))

    return similarity_matrix

from sklearn.decomposition import PCA
gaussian_sim = calculate_gaussian_similarity(association_matrix)
pca = PCA(n_components=396)
emb = pca.fit_transform(gaussian_sim)

with open('source/Gaussian_sim_LncRNA.txt', 'w') as file:
    for row in emb:
        file.write(' '.join(map(str, row)) + '\n')

gaussian_df = pd.DataFrame(gaussian_sim, index=miRNA_names, columns=miRNA_names)
print("\n高斯相似性矩阵:")
print(gaussian_df)


import matplotlib.pyplot as plt
import seaborn as sns

def plot_similarity_matrix(sim_matrix, title, miRNA_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(sim_matrix, annot=True, cmap='YlOrRd',
                xticklabels=miRNA_names, yticklabels=miRNA_names)
    plt.title(title)
    plt.show()

#plot_similarity_matrix(gaussian_sim, "miRNA功能相似性 (Gaussian)", miRNA_names)
#高斯核方法可以捕捉非线性关系
