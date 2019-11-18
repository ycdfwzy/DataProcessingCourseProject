# clustering method:
# Partitioning method -> k-means, k-medoids
# Hierarchical method
# Density-based method -> DBSCAN, SNN

import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

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

    # KMeans
    # kmeas_model = KMeans(n_clusters=4, random_state=1).fit(data_x)
    # labels = kmeas_model.labels_
    # print('Calinski-Harabasz:', metrics.calinski_harabasz_score(data_x, labels))
    # print('Silhouette Coefficient:', metrics.silhouette_score(data_x, labels, metric='euclidean'))

    # DBSCAN
    db = DBSCAN(eps=8, min_samples=20).fit(data_x)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print('Calinski-Harabasz:', metrics.calinski_harabasz_score(data_x, labels))
    print('Silhouette Coefficient:', metrics.silhouette_score(data_x, labels, metric='euclidean'))