# -*- coding: utf-8 -*-
# @Time    : 2019/9/18 0:04
# @Author  : Weiyang
# @File    : XGBoostClassifier.py

#==================================
# XGBoost用于分类的类
#==================================

from XGBoost import XGBoost
import numpy as np

class XGBoostClassifier(XGBoost):
    '''XGBoost分类器'''

    def __init__(self, n_estimators=200,  min_samples_split=2,
                 min_impurity=1e-7, max_depth=float("inf"), isRegression=False, lambd=1, gama=0,
                 objective="logistic"):
        super(XGBoostClassifier,self).__init__(n_estimators=n_estimators,
                                               min_samples_split=min_samples_split,
                                               min_impurity=min_impurity,max_depth=max_depth,
                                               isRegression=isRegression,lambd=lambd,gama=gama,
                                               objective=objective)

if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    # 计算准确率
    def accuracy_score(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
        return accuracy

    print('--------XGBoost Classifier --------')
    data = datasets.load_iris()
    X = data.data
    Y = data.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)

    # 二分类，我们将鸢尾花中类别为1,2的样本，设为label=1；类别为0的样本，label仍为0
    print("二分类结果：")
    model = XGBoostClassifier(objective='logistic')
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    # 将鸢尾花中类别为1,2的样本，设为label=1；类别为0的样本，label仍为0
    temp = []
    for label in Y_test:
        if label == 0:
            temp.append(0)
        else:
            temp.append(1)
    Y_test = np.array(temp)
    accuracy = accuracy_score(Y_test, y_pred)
    print("    Accuracy:", accuracy)
    '''

    # 多分类，鸢尾花中类别有三类，分别是0,1,2
    print("多分类结果：")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)
    model = XGBoostClassifier(objective='crossEntropy')
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    print("    Accuracy:", accuracy)
    '''