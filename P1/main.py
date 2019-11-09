import time
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler


def classification(X, y):
    # clf = KNeighborsClassifier(2)
    # clf = tree.DecisionTreeClassifier()
    # clf = svm.LinearSVC()
    # clf = MultinomialNB(alpha=0.5)
    clf = MLPClassifier([160, 60, 6], learning_rate_init=0.001, activation='relu', \
                        solver='adam', alpha=0.0001, max_iter=30000)
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    return scores.mean()

if __name__ == '__main__':
    with open('../data/classification_data.csv') as fin:
        csv_data = csv.reader(fin)
        data_x = []
        data_y = []
        flag = True
        for row in csv_data:
            if flag:
                flag = False
                continue
            raw_data = [int(row[1 + i]) for i in range(len(row[1:-1]))]
            t = min(raw_data)
            data_x.append([x for x in raw_data])
            data_y.append(int(row[-1]))

        scaler = StandardScaler(copy=False)
        np_data_x = np.array(data_x)
        np_data_y = np.array(data_y)
        np_data_x = scaler.fit_transform(np_data_x)

        start = time.time()
        print(classification(np_data_x, np_data_y))
        end = time.time()
        print('totally cost for 5-classification: ', end - start)

        np_data_y = np.array([1 if y == 1 else 0 for y in data_y])
        start = time.time()
        print(classification(np_data_x, np_data_y))
        end = time.time()
        print('totally cost for 2-classification: ', end - start)

