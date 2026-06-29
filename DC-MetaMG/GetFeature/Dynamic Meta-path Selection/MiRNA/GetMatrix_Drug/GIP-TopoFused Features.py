import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from sklearn.metrics import jaccard_score
from sklearn.decomposition import PCA

# 示例数据 - 实际应用中替换为你的数据
miRNA_names = []
disease_names = []


with open('drug.txt', 'r') as infile:
    for line in infile:
        line = line.strip().split('\t')
        miRNA_names.append(line[1])


with open('Disease.txt', 'r') as infile:
    for line in infile:
        line = line.strip().split('\t')
        disease_names.append(line[1])



#file_path = 'array_MD.txt'
file_path = 'DD_ARR.txt'

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


gaussian_sim = calculate_gaussian_similarity(association_matrix)



from node2vec import Node2Vec
import networkx as nx

num_drugs = 60
num_diseases = 1937
associations = association_matrix

drug_nodes = [f"drug_{i}" for i in range(num_drugs)]
disease_nodes = [f"disease_{i}" for i in range(num_diseases)]

G = nx.Graph()

# 添加二分节点（注意bipartite属性）
G.add_nodes_from(drug_nodes, bipartite=0)    # 药物节点属于分区0
G.add_nodes_from(disease_nodes, bipartite=1) # 疾病节点属于分区1

# 添加边（仅当关联存在时）
for i, drug in enumerate(drug_nodes):
    for j, disease in enumerate(disease_nodes):
        if associations[i, j] == 1:
            G.add_edge(drug, disease)

# 参数说明：
# dimensions: 嵌入维度 (推荐128-512)
# walk_length: 每条随机游走的长度
# num_walks: 每个节点的游走次数
# p: 返回参数 (1.0为无偏随机游走)
# q: 探索参数 (1.0为BFS和DFS平衡)
node2vec = Node2Vec(
    G,
    #dimensions=138,      # 嵌入维度
    dimensions=198,      # 嵌入维度
    walk_length=30,      # 每次游走30步
    num_walks=200,       # 每个节点游走200次
    p=1,                 # 返回参数
    q=1,                 # 探索参数
    workers=4,           # 并行线程数
    weight_key=None      # 如果边有权重，指定属性名
)

# 训练模型 (Skip-gram)
model = node2vec.fit(
    window=10,           # 上下文窗口大小
    min_count=1,         # 忽略出现次数低于此值的节点
    batch_words=10000,   # 每个batch处理的词数
    seed=42              # 随机种子
)

drug_embeddings = {drug: model.wv[drug] for drug in drug_nodes}

# 转换为矩阵形式 (60 drugs × 256 dims)
drug_embedding_matrix = np.array([drug_embeddings[drug] for drug in drug_nodes])

# 保存嵌入结果
pd.DataFrame(drug_embedding_matrix, index=drug_nodes).to_csv("drug_embeddings.csv")

# ======================
# 5. 可视化 (PCA降维)
# ======================

'''
# 降维到长度2
pca = PCA(n_components=2)
embed_2d = pca.fit_transform(drug_embedding_matrix)'''




'''
with open('source/GIP_Topo_drug.txt', 'w') as file:
    for row, row1 in zip(gaussian_sim, drug_embedding_matrix):
        # 将两个矩阵的行合并为一个列表

        # 将合并后的行写入文件
        file.write(' '.join(map(str, row)) + ' ')
        file.write(' '.join(map(str, row1)) + '\n')'''

with open('source/Topo_drug.txt', 'w') as file:
    for row in drug_embedding_matrix:
        # 将两个矩阵的行合并为一个列表

        # 将合并后的行写入文件
        file.write(' '.join(map(str, row)) + '\n')





gaussian_df = pd.DataFrame(gaussian_sim, index=miRNA_names, columns=miRNA_names)
print("\n高斯相似性矩阵:")
print(gaussian_df)



'''
import matplotlib.pyplot as plt
import seaborn as sns

def plot_similarity_matrix(sim_matrix, title, miRNA_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(sim_matrix, annot=True, cmap='YlOrRd',
                xticklabels=miRNA_names, yticklabels=miRNA_names)
    plt.title(title)
    plt.show()

#plot_similarity_matrix(gaussian_sim, "miRNA功能相似性 (Gaussian)", miRNA_names)
#高斯核方法可以捕捉非线性关系'''
