# -*- coding: utf-8 -*-
# @Time    : 2019/9/22 15:04
# @Author  : Weiyang
# @File    : compare.py

#==========================================================
# 比较kd树与线性扫描实现的KNN
#==========================================================

from KNN_linearScan import KNN_linearScan
from KNN_kdTree import KNN_kdTree
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import math

# 计算准确率
def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

def euclidean_distance(x1, x2):
    """ Calculates the l2 distance between two vectors """
    distance = 0
    # Squared distance between each coordinate
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)

data = datasets.load_iris()
X = data.data
Y = data.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

print('--------KNN KDTree Classification  --------')
model = KNN_kdTree(K=5)
model.fit(X_train,Y_train)
y_pred,k1 = model.predict(X_test,K=5,isRegression=False)
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)

print('--------KNN Linear Scan Classification  --------')
model = KNN_linearScan(K=5)
y_pred,k2 = model.predict(X_test,X_train,Y_train,isRegression=False)
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)

print('比较KD树实现的KNN的最近邻点，与线性扫描实现的KNN最近邻点的差异')
# 随机选择一个样例查看
num = np.random.randint(len(X_test))
print('kd树的最近邻点，以及相应的距离')
print(np.array([list(i) for i in k1[num]]))
for vec in k1[num]:
    vec = np.array(vec)
    print(euclidean_distance(X_test[num],vec))
print()
print('线性扫描的最近邻点，以及相应的距离')
print(k2[num])
for vec in k2[num]:
    vec = np.array(vec)
    print(euclidean_distance(X_test[num],vec))

