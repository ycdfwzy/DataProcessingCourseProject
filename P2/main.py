# clustering method:
# Partitioning method -> k-means, k-medoids
# Hierarchical method
# Density-based method -> DBSCAN, SNN

import numpy as np
from sklearn import metrics
from clusters.KMeans import KMeans
from clusters.Spectral import SpectralClustering
from clusters.Hierarchical import HierarchicalClustering
import matplotlib.pyplot as plt

def read_data():
    data_x = []
    with open('data/cluster_data.txt') as fin:
        lines = fin.read().split('\n')
        for line in lines:
            if line == '':
                continue
            raw_data = [int(num) for num in line.split(' ')]
            data_x.append(raw_data)
    return data_x

if __name__ == "__main__":
    data = read_data()
    data_x = np.array(data)

    # kmeans_model = KMeans(5, random_state=4) # 4 and 6 can get pretty good result
    # kmeans_model.fit(data_x)
    # labels = kmeans_model.label_pred

    # spec = SpectralClustering(5, gamma=0.01, random_state=4)
    # spec.fit(data_x)
    # labels = spec.label_pred

    heir = HierarchicalClustering(5)
    heir.fit(data_x)
    labels = heir.label_pred

    print('Calinski-Harabasz:', metrics.calinski_harabasz_score(data_x, labels))
    print('Silhouette Coefficient:', metrics.silhouette_score(data_x, labels, metric='euclidean'))
