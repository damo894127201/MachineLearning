# -*- coding: utf-8 -*-
# @Time    : 2019/9/14 0:00
# @Author  : Weiyang
# @File    : test.py

import numpy as np

def labelCounts( Y):
    '''统计各个类别的取值个数,Y=[label1,label2,...]'''
    results = {}
    for label in Y:
        # 判断label是否在results
        if label not in results:
            results[label] = 0
        results[label] += 1  # 计数加1
    return results

def calculate_gini( y, y1, y2):
    '''计算Gini指数，值越小越好，Gini指数是不确定性的度量'''
    # 计算子集y1的gini指数
    label_count1 = labelCounts(y1)  # 子集y1中各个类别的数量
    y1_gini = 1.0
    for label in label_count1.keys():
        y1_gini -= (float(label_count1[label]) / len(y1)) ** 2
    # 计算子集y2的gini指数
    label_count2 = labelCounts(y2)  # 子集y2中各个类别的数量
    y2_gini = 1.0
    for label in label_count2.keys():
        y2_gini -= (float(label_count2[label]) / len(y2)) ** 2
    # 计算集合y关于当前特征的Gini指数
    p1 = float(len(y1)) / len(y)
    p2 = float(len(y2)) / len(y)
    gini = p1 * y1_gini + p2 * y2_gini
    # 由于在DecisionTree类中调用时，采用的是值越大越好的策略，因此需要将gini添上负号
    return gini

y = np.array([0,0,1,1,0,0,0,1,1,1,1,1,1,1,0])
y1 = np.array([0,0,1,1,0])
y2 = np.array([0,0,1,1,1,1,1,1,1,0])
print(calculate_gini(y,y1,y2))