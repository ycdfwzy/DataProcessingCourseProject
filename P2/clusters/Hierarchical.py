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

    def __traverse(self, node, label):
        if node.left is None and node.right is None:
            self.label_pred[node.id] = label
        if node.left:
            self.__traverse(node.left, label)
        if node.right:
            self.__traverse(node.right, label)

    def __set_labels(self, tree):
        for i in range(self.__n_clusters):
            self.__traverse(tree[i], i)

    def __cluster(self):
        m, n = np.shape(self.__dataset)
        distance = dict()
        nodes = [ClusterNode(self.__dataset[i], id=i) for i in range(m)]
        cluster_id = -1

        # cluster process
        while len(nodes) > self.__n_clusters:
            min_dis = float("inf")
            length = len(nodes)
            closest = None
            
            # 寻找距离最近的两个聚类
            for i in range(length - 1):
                for j in range(i + 1, length):
                    key = (nodes[i].id, nodes[j].id)
                    if key not in distance:
                        distance[key] = self.__distance(nodes[i].center, nodes[j].center)
                    if distance[key] < min_dis:
                        min_dis = distance[key]
                        closest = (i, j)

            # 合并两个聚类
            node1, node2 = nodes[closest[0]], nodes[closest[1]]
            new_point = np.array([(
                node1.center[i] * node1.count + node2.center[i] * node2.count) / (node1.count + node2.count) for i in range(n)])
            new_node = ClusterNode(new_point, 
                                   left=node1, 
                                   right=node2,
                                   distance=min_dis,
                                   id=cluster_id,
                                   count=node1.count + node2.count)
            cluster_id -= 1
            del nodes[closest[1]], nodes[closest[0]]
            nodes.append(new_node)
            print('done for %d' % cluster_id)

        self.label_pred = [-1] * m
        self.__set_labels(nodes)

    def fit(self, dataset):
        self.__dataset = dataset
        self.__cluster()
        return