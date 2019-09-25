# -*- coding: utf-8 -*-
# @Time    : 2019/9/25 20:15
# @Author  : Weiyang
# @File    : SVMClassification.py

#=======================================================================
# SVM 用于分类
#=======================================================================

from SVM import SVM
import numpy as np
from sklearn.datasets import make_classification
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 计算准确率
def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X  对各个特征向量进行归一化，消除量纲影响"""
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1  # 防止模长为0
    return X / np.expand_dims(l2, axis)

# 构造数据
X, y = make_classification(n_samples=100, n_features=10, n_informative=5,
                               random_state=1111, n_classes=2, class_sep=1.75, )
# y的标签取值{0,1} 变成 {-1, 1}
y = (y * 2) - 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# 对数据归一化，消除量纲影响
# X_train ,X_test = normalize(X_train),normalize(X_test)

# 构造SVM分类器
model = SVM(X_train,y_train,kernel="RBF")
model.fit()
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
#print(y_pred,y_test)
print("随机产生的分类数据的 Accuracy:", accuracy)
print()

print('--------------------------------------------------------------')
print()
# 二分类，我们将鸢尾花中类别为1,2的样本，设为label=1；类别为0的样本，label为-1
data = datasets.load_iris()
X = data.data
Y = data.target
temp = []
for label in Y:
    if label == 0:
        temp.append(-1)
    else:
        temp.append(1)
Y = np.array(temp)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)

# 构造SVM分类器
model = SVM(X_train,y_train,kernel="RBF")
model.fit()
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("鸢尾花数据的  Accuracy:", accuracy)




