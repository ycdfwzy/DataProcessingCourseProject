# clustering method:
# Partitioning method -> k-means, k-medoids
# Hierarchical method
# Density-based method -> DBSCAN, SNN

import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AgglomerativeClustering

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
    print("KMeans")
    kmeas_model = KMeans(n_clusters=5, random_state=1).fit(data_x)
    labels = kmeas_model.labels_
    print('Calinski-Harabasz:', metrics.calinski_harabasz_score(data_x, labels))
    print('Silhouette Coefficient:', metrics.silhouette_score(data_x, labels, metric='euclidean'))

    # DBSCAN
    print("\nDBSCAN")
    db = DBSCAN(eps=8, min_samples=10).fit(data_x)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print('Calinski-Harabasz:', metrics.calinski_harabasz_score(data_x, labels))
    print('Silhouette Coefficient:', metrics.silhouette_score(data_x, labels, metric='euclidean'))

    # Birch
    print("\nBirch")
    birch = Birch(n_clusters=5).fit(data_x)
    labels = birch.labels_
    print('Calinski-Harabasz:', metrics.calinski_harabasz_score(data_x, labels))
    print('Silhouette Coefficient:', metrics.silhouette_score(data_x, labels, metric='euclidean'))

    # SpectralClustering gamma: 0.01 -> 5
    print("\nSpectralClustering")
    spec = SpectralClustering(n_clusters=5, gamma=0.03).fit(data_x)
    labels = spec.labels_
    print('Calinski-Harabasz:', metrics.calinski_harabasz_score(data_x, labels))
    print('Silhouette Coefficient:', metrics.silhouette_score(data_x, labels, metric='euclidean'))

    # MeanShift
    print("\nMeanShift")
    bandwidth = estimate_bandwidth(data_x, quantile=0.16, n_samples=None)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(data_x)
    labels = ms.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Calinski-Harabasz:', metrics.calinski_harabasz_score(data_x, labels))
    print('Silhouette Coefficient:', metrics.silhouette_score(data_x, labels, metric='euclidean'))

    # Agglomerative Clustering
    print("\nAgglomerative Clustering")
    for linkage in ('ward', 'average', 'complete', 'single'):
        clustering = AgglomerativeClustering(linkage=linkage, n_clusters=5)
        agg = clustering.fit(data_x)
        labels = agg.labels_
        print(linkage)
        print('Calinski-Harabasz:', metrics.calinski_harabasz_score(data_x, labels))
        print('Silhouette Coefficient:', metrics.silhouette_score(data_x, labels, metric='euclidean'))
