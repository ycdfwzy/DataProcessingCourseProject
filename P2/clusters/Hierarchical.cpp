#include<iostream>
#include<vector>
#include<map>
#include<list>
#include<math.h>
#include<climits>

using namespace std;

class ClusterNode {
public:
    vector<double> center;
    ClusterNode* left;
    ClusterNode* right;
    double distance;
    int id;
    int count;
    ClusterNode(vector<double> center, int id = INT_MIN, ClusterNode* left = nullptr, 
        ClusterNode* right = nullptr, double distance = -1, int count = 1) {
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
        if (node.left) traverse(*node.right, label);
        if (node.right) traverse(*node.right, label);
    }

    void set_labels(list<ClusterNode> tree) {
        int label = 0;
        for (auto p = tree.begin(); p != tree.end(); p++)
            traverse(*p, label);
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

        // while (nodes.size() > n_clusters) {
        //     double min_dis = INFINITY;
        //     int length = nodes.size();
        //     pair<int, int> clostest;
        //     for (int i = 0; i < length - 1; i++)
        //         for (int j = i + 1; j < length; j++) {
        //             pair<int ,int> key(nodes[i].id, nodes[j].id);
        //             auto p = distance.find(pair<int, int>(, j));
        //             double dis = 0;
        //             if (p != distance.end()) dis = p->second;
        //             else distance.insert(make_pair(pair<int, int>(i, j)))
        //         }
            
        // }
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
