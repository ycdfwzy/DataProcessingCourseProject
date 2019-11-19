# coding = utf8

import numpy as np
from .KMeans import KMeans

class SpectralClustering:
    def __init__(self, n_clusters, gamma=1.0):
        self.__n_clusters = n_clusters
        self.__dataset = None
        self.__gamma = gamma

    def __distance(self, x, y):
        # 欧氏距离
        return np.sqrt(np.sum((x - y) ** 2))

    def __distance_matrix(self):
        m = np.shape(self.__dataset)[0]
        dis_mat = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                dis_mat[i][j] = self.__distance(self.__dataset[i], self.__dataset[j])
                dis_mat[j][i] = dis_mat[i][j]
        return dis_mat

    def __affinity_matrix(self, dis_mat):
        m = np.shape(self.__dataset)[0]
        aff_mat = np.zeros((m, m))

        for i in range(m):
            dis_ind = zip(dis_mat[i], range(m))
            dis_ind = sorted(dis_ind, key=lambda x:x[0])
            neighbours = [dis_ind[j][1] for j in range(self.__n_clusters + 1)] # nearest k neighbours

            for j in neighbours:
                aff_mat[i][j] = aff_mat[j][i] = np.exp(-self.__gamma * dis_mat[i][j] ** 2)

        return aff_mat

    def __laplacian_matrix(self, aff_mat):
        # degree matrix
        degree_mat = np.sum(aff_mat, axis=1)

        # laplacian matrix
        laplacian_mat = np.diag(degree_mat) - aff_mat

        # normalize
        norm_degree = np.diag(1.0 / (degree_mat ** 0.5))
        return np.dot(np.dot(norm_degree, laplacian_mat), norm_degree)

    def __cluster(self):
        dis_mat = self.__distance_matrix()
        aff_mat = self.__affinity_matrix(dis_mat)
        lap_mat = self.__laplacian_matrix(aff_mat)
        lam, H = np.linalg.eig(lap_mat)
        kmeas_model = KMeans(self.__n_clusters, random_state=4)
        kmeas_model.fit(H)
        return kmeas_model.centroids, kmeas_model.label_pred

    def fit(self, dateset):
        self.__dataset = dateset
        self.centroids, self.label_pred = self.__cluster()
        return
