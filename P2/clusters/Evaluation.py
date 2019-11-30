# Intra-cluster cohesion and Inter-cluster separation

import numpy as np

def euclid_distance(x, y):
    return np.sum((x - y) ** 2)

# 类内凝聚度，越低越好
def intra_cluster_cohesion(X, labels):
    assert len(X) == len(labels), "data length not the same with labels length"
    clusters = dict()
    for i in range(len(X)):
        if labels[i] not in clusters:
            clusters[labels[i]] = []
        clusters[labels[i]].append(X[i])
    
    wss = 0
    for key, val in clusters.items():
        mean = np.average(val, axis=0)
        for p in val:
            wss += euclid_distance(p, mean)
    return wss

# 类间分离度，越高越好
def inter_cluster_separation(X, labels):
    assert len(X) == len(labels), "data length not the same with labels length"
    clusters = dict()
    for i in range(len(X)):
        if labels[i] not in clusters:
            clusters[labels[i]] = []
        clusters[labels[i]].append(X[i])

    bss = 0
    all_mean = np.average(X)
    for key, val in clusters.items():
        mean = np.average(val, axis=0)
        bss += len(val) * euclid_distance(all_mean, mean)
    return bss