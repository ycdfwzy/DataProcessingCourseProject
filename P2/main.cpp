#include<iostream>
#include<fstream>
#include<vector>
#include"clusters/Hierarchical.hpp"

using namespace std;

int main() {
    int m = 2000;
    int n = 7;
    vector<vector<double>> data_x(m, vector<double>(n, 0));

    // 读取数据
    ifstream infile;
    infile.open("./data/cluster_data.txt", ios::in);
    ofstream label_file;
    label_file.open("./data/labels.txt", ios::out | ios::trunc);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            infile >> data_x[i][j];

    // 进行层次聚类
    HierarchicalClustering hier(5);
    hier.fit(data_x);
    // 将结果写回文件
    for (int i = 0; i < m; i++)
        label_file << hier.label_pred[i] << endl;
    
    infile.close();
    label_file.close();
    return 0;
}