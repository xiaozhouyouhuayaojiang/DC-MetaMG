import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from sklearn.metrics import jaccard_score
from sklearn.decomposition import PCA


miRNA_names = []
disease_names = []


with open('Drug_ID.txt', 'r') as infile:
    for line in infile:
        line = line.strip().split('\t')
        miRNA_names.append(line[1])


with open('Disease_ID.txt', 'r') as infile:
    for line in infile:
        line = line.strip().split('\t')
        disease_names.append(line[1])



#file_path = 'array_MD.txt'
file_path = 'DD_ARR.txt'


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


gaussian_sim = calculate_gaussian_similarity(association_matrix)



from node2vec import Node2Vec
import networkx as nx

num_drugs = 154
num_diseases = 403
associations = association_matrix

drug_nodes = [f"drug_{i}" for i in range(num_drugs)]
disease_nodes = [f"disease_{i}" for i in range(num_diseases)]

G = nx.Graph()


G.add_nodes_from(drug_nodes, bipartite=0)
G.add_nodes_from(disease_nodes, bipartite=1)

for i, drug in enumerate(drug_nodes):
    for j, disease in enumerate(disease_nodes):
        if associations[i, j] == 1:
            G.add_edge(drug, disease)


node2vec = Node2Vec(
    G,
    #dimensions=138,
    dimensions=396,
    walk_length=30,
    num_walks=200,
    p=1,
    q=1,
    workers=4,
    weight_key=None
)


model = node2vec.fit(
    window=10,
    min_count=1,
    batch_words=10000,
    seed=42
)

drug_embeddings = {drug: model.wv[drug] for drug in drug_nodes}

drug_embedding_matrix = np.array([drug_embeddings[drug] for drug in drug_nodes])


pd.DataFrame(drug_embedding_matrix, index=drug_nodes).to_csv("drug_embeddings.csv")


with open('source/Topo_drug.txt', 'w') as file:
    for row in drug_embedding_matrix:

        file.write(' '.join(map(str, row)) + '\n')


gaussian_df = pd.DataFrame(gaussian_sim, index=miRNA_names, columns=miRNA_names)

print(gaussian_df)

