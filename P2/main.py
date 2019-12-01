# clustering method:
# Partitioning method -> k-means, k-medoids
# Hierarchical method
# Density-based method -> DBSCAN, SNN

import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from clusters.KMeans import KMeans
from clusters.Spectral import SpectralClustering
from clusters.Hierarchical import HierarchicalClustering
from clusters import Evaluation, Visualization
import matplotlib.pyplot as plt

# 读取聚类数据
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

# 读取C++写入的label
def read_label():
    labels = []
    with open("data/labels.txt") as fin:
        for row in fin.read().split('\n'):
            if row == '':
                continue
            labels.append(int(row))
    return labels

if __name__ == "__main__":
    data = read_data()
    data_x = np.array(data)
    
    # K Means聚类
    # kmeans_model = KMeans(5, random_state=4) # 4 and 6 can get pretty good result
    # kmeans_model.fit(data_x)
    # labels = kmeans_model.label_pred

    # 谱聚类
    # spec = SpectralClustering(5, gamma=0.01, random_state=4)
    # spec.fit(data_x)
    # labels = spec.label_pred

    # 层次聚类
    # heir = HierarchicalClustering(5)
    # heir.fit(data_x)
    # labels = heir.label_pred

    labels = read_label()

    # 数据分析
    print('Calinski-Harabasz:', metrics.calinski_harabasz_score(data_x, labels))
    print('Silhouette Coefficient:', metrics.silhouette_score(data_x, labels, metric='euclidean'))
    print('Intra-cluster Cohesion:', Evaluation.intra_cluster_cohesion(data_x, labels))
    print('Inter-cluster Separation', Evaluation.inter_cluster_separation(data_x, labels))

    # 可视化
    Visualization.TSNE_scatter(data_x, labels)
    # Visualization.PCA_scatter(data_x, labels)
    # Visualization.Factor_scatter(data_x, labels)
