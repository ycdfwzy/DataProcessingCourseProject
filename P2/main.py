# clustering method:
# Partitioning method -> k-means, k-medoids
# Hierarchical method
# Density-based method -> DBSCAN, SNN

import numpy as np
from sklearn import metrics
from clusters.KMeans import KMeans
from clusters.Spectral import SpectralClustering
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
    cal_score = []
    sil_score = []
    for gamma in (0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5):
        spec = SpectralClustering(5, gamma=gamma)
        spec.fit(data_x)
        labels = spec.label_pred
        cal_score.append(metrics.calinski_harabasz_score(data_x, labels))
        sil_score.append(metrics.silhouette_score(data_x, labels, metric='euclidean'))
    plt.figure(1)
    plt.plot(cal_score)
    plt.show()
    plt.figure(2)
    plt.plot(sil_score)
    plt.show()
    # print('Calinski-Harabasz:', metrics.calinski_harabasz_score(data_x, labels))
    # print('Silhouette Coefficient:', metrics.silhouette_score(data_x, labels, metric='euclidean'))
