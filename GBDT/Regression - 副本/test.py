# -*- coding: utf-8 -*-
# @Time    : 2019/9/15 23:33
# @Author  : Weiyang
# @File    : test.py
import numpy as np

def calculate_variance(X):
    """ Return the variance of the features in dataset X """
    mean = np.ones(np.shape(X)) * X.mean(0)
    n_samples = np.shape(X)[0]
    variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))

    return variance

def calculate_variance2(Y):
    '''
    计算平方误差sum((y-y_pred)^2)/N=Var(y),
    Y = np.array([[value1,value2,...],...])或Y = np.array([value,...])
    Y可能是单维的，也可以是多维的
    '''
    import numpy as np
    #Y = np.squeeze(Y)
    return np.var(Y,axis=0)

t1 = np.array([[1,2,3],[4,5,6],[6,7,8]])
print(calculate_variance(t1))
print(calculate_variance2(t1))
print(np.ones(np.shape(t1)) * t1.mean(0))
