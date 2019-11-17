#coding: utf-8

import numpy as np
import pandas as pd
import random

class DecisionTree:
    def __init__(self):
        pass

    def __cal_entropy(data, labels):
        label_count = {}
        for label in labels:
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
        entropy = 0.
        

    def __build_decision_tree(self):
        pass

    def __choose_best_feature(self, data):
        count_feature = len(data[0])


    