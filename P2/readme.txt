本文件夹中包含了问题2的源代码文件

使用`python3 main.py`即可运行算法，并进行可视化
main.cpp是为层次聚类写的C++文件，使用`g++ -std=c++11 main.cpp`即可编译
sklearn_test.py是使用sklearn进行测试的文件，对sklearn中的多种聚类算法进行了验证和对比

cluster文件夹中包含了聚类算法和可视化、验证方法的实现，包含了K Means，层次聚类，谱聚类这几种方法，评估方面自行编写了类间凝聚度和类内分离度

data文件夹中存放计算的源数据和聚类结果