#include<iostream>
#include<vector>
#include<map>
#include<list>
#include<math.h>
#include<iterator>
#include<climits>

using namespace std;

class ClusterNode {
public:
    vector<double> center;
    shared_ptr<ClusterNode> left;
    shared_ptr<ClusterNode> right;
    // ClusterNode* left;
    // ClusterNode* right;
    double distance;
    int id;
    int count;
    ClusterNode(vector<double> center, int id = INT_MIN, shared_ptr<ClusterNode> left = nullptr, 
        shared_ptr<ClusterNode> right = nullptr, double distance = -1, int count = 1) {
        this -> center = center;
        this -> left = left;
        this -> right = right;
        this -> distance = distance;
        this -> id = id;
        this -> count = count;
    }
    ~ClusterNode() {}
};

class HierarchicalClustering {
private:
    int n_clusters;
    vector<vector<double>> dataset;
    int m, n;
public:
    vector<int> label_pred;

private:
    double distance(vector<double> x, vector<double> y) {
        double sum = 0;
        for (int i = 0; i < this -> n; i++)
            sum += (x[i] - y[i]) * (x[i] - y[i]);
        return sqrt(sum);
    }

    void traverse(ClusterNode node, int label) {
        if (node.left == nullptr && node.right == nullptr)
            this -> label_pred[node.id] = label;
        if (node.left) traverse(*node.left, label);
        if (node.right) traverse(*node.right, label);
    }

    void set_labels(list<ClusterNode> tree) {
        int label = 0;
        for (auto p = tree.begin(); p != tree.end(); p++) {
            traverse(*p, label);
            label++;
        }
    }

    void cluster() {
        m = dataset.size();
        n = dataset[0].size();
        map<pair<int, int>, double> distance;
        list<ClusterNode> nodes;
        for (int i = 0; i < m; i++) {
            nodes.push_back(ClusterNode(dataset[i], i));
        }
        int cluster_id = -1;

        while (nodes.size() > n_clusters) {
            double min_dis = INFINITY;
            pair<list<ClusterNode>::iterator, list<ClusterNode>::iterator> clostest;

            // 遍历寻找距离最近的两个聚类
            for (auto p = nodes.begin(); p != prev(nodes.end()); p++)
                for (auto q = next(p); q != nodes.end(); q++) {
                    pair<int ,int> key(p->id, q->id);
                    auto it = distance.find(key);
                    double dis = 0;
                    if (it != distance.end()) dis = it->second;
                    else {
                        dis = this->distance(p->center, q->center);
                        distance.insert(make_pair(key, dis));
                    }
                    if (dis < min_dis) { min_dis = dis; clostest = make_pair(p, q);}
                }
            
            // 合并两个聚类
            vector<double> new_point(n, 0);
            for (int i = 0; i < n; i++)
                new_point[i] = (clostest.first->center[i] * clostest.first->count + clostest.second->center[i] * clostest.second->count) / 
                    (clostest.first->count + clostest.second->count);
            
            ClusterNode new_node(new_point, cluster_id, shared_ptr<ClusterNode>(&(*clostest.first)),
                shared_ptr<ClusterNode>(&(*clostest.second)), min_dis, clostest.first->count + clostest.second->count);
            cluster_id--;
            nodes.erase(clostest.first);
            nodes.erase(clostest.second);
            nodes.push_back(new_node);
            printf("done for %d\n", cluster_id);
        }

        label_pred = vector<int>(m, -1);
        this->set_labels(nodes);
    }
public:
    HierarchicalClustering(int n_clusters) {
        this -> n_clusters =  n_clusters;
    }
    ~HierarchicalClustering() {}

    void fit(vector<vector<double>> dataset) {
        this->dataset = dataset;
        cluster();
    }
};
