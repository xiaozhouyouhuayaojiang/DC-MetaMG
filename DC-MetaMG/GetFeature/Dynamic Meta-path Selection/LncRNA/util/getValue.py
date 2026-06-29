from sklearn.cluster import KMeans
import numpy as np

def getValue(filename):
    association_matrix = []
    with open(filename, 'r') as file:
        for line in file:

            row = line.strip().split(' ')

            association_matrix.append(list(map(float, row)))

    association_matrix = np.array(association_matrix, dtype=float)
    theta_kmeans = find_theta_kmeans(association_matrix)
    return theta_kmeans
def find_theta_kmeans(matrix, n_clusters=2):
    values = matrix[np.triu_indices_from(matrix, k=1)].reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters).fit(values)
    return np.mean(kmeans.cluster_centers_)


