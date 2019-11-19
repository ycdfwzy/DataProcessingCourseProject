# coding = utf8

import numpy as np

class ClusterNode:
    def __init__(self, center, left=None, right=None, distance=-1, id=None, count=1):
        self.center = center
        self.left = left
        self.right = right
        self.distance= distance
        self.id = id
        self.count = count

class HierarchicalClustering:
    def __init__(self, n_clusters):
        self.__n_clusters = n_clusters
        self.__dataset = None
    
    def __distance(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def fit(self, dataset):
        pass