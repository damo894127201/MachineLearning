# -*- coding: utf-8 -*-
# @Time    : 2019/9/14 15:43
# @Author  : Weiyang
# @File    : testVariance.py
import numpy as np

def calculate_variance(X):
    """ Return the variance of the features in dataset X """
    mean = np.ones(np.shape(X)) * X.mean(0)
    n_samples = np.shape(X)[0]
    variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))

    return variance

def calculate_variance2(Y):
    '''计算平方误差sum((y-y_pred)^2)/N=Var(y),Y = np.array([value1,value2,...])'''
    import numpy as np
    Y = np.squeeze(Y)
    return np.sum(np.var(Y))

y = np.array([[1],[2],[1],[0],[2]])
print(y.shape)
print(calculate_variance(y))
print(calculate_variance2(y))
