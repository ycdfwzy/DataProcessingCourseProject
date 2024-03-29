# coding = utf8

import numpy as np

class KMeans:
    def __init__(self, n_clusters, random_state=None):
        self.__n_clusters = n_clusters
        self.__dataset = None
        self.__random_state = random_state

    def __distance(self, x, y):
        # 欧氏距离
        return np.sqrt(np.sum((x - y) ** 2))

    def __rand_centers(self):
        m, n = self.__dataset.shape
        centroids = np.zeros((self.__n_clusters, n))
        indexs = []

        # get random generator
        rdm = None
        try:
            rdm = np.random.RandomState(self.__random_state)
        except:
            rdm = np.random

        # 随机获取聚类初始中心点
        for i in range(self.__n_clusters):
            ind = int(rdm.uniform(0, m))
            while ind in indexs:
                ind = int(rdm.uniform(0, m))
            indexs.append(ind)
            centroids[i, :] = self.__dataset[ind, :]
        return centroids
    
    def __cluster(self):
        m = np.shape(self.__dataset)[0]
        cluster_assign = [0.] * m
        centroids = self.__rand_centers()
        assign_changed = True
        
        while assign_changed:
            assign_changed = False

            for i in range(m):
                min_dis = float("inf")
                min_ind = -1

                # 寻找距离最近的簇
                for j in range(self.__n_clusters):
                    dis = self.__distance(centroids[j, :], self.__dataset[i, :])
                    if dis < min_dis:
                        min_dis = dis
                        min_ind = j
                
                # 更新样本所属的簇
                if cluster_assign[i] != min_ind:
                    assign_changed = True
                    cluster_assign[i] = min_ind
            
            # 更新聚类样本的簇中心
            for j in range(self.__n_clusters):
                cluster_points = self.__dataset[np.nonzero(np.array(cluster_assign) == j)[0]]
                centroids[j, :] = np.mean(cluster_points, axis=0)
        
        return centroids, cluster_assign

    def fit(self, dataset):
        self.__dataset = dataset
        self.centroids, self.label_pred = self.__cluster()
        return
