# -*- coding: utf-8 -*-
# @Time    : 2019/9/17 19:11
# @Author  : Weiyang
# @File    : LossFunction.py

#=================================================================================================
# XGBoost的损失函数：
# 损失函数可以自定义，但需要给出损失函数对 y_pred (= f_(t-1)(x)) 的一阶导数和二阶导数的计算方式,
#                    注意这里的y_pred是之前所有回归树预测的值累加

# 本模块实现了 最小二乘损失(二乘，又叫平方，最小平方损失，即均方差损失MSE)和逻辑损失的一阶导数和二阶导数

# 最小二乘法(Least Squares Method)不是一种算法，而是一种损失函数的模型。用最小二乘法表示损失函数后
# 再通过其它优化算法求解最优值
# 流程：
# 1. 损失函数：loss_function = 1/2 * sum((真实值-预测值)^2)
# 2. 解法：求偏导等于0的极值点
#=================================================================================================

import numpy as np

class LeastSquareLoss:
    '''最小二乘损失,均方差损失'''
    def g(self,y_true,y_pred):
        '''一阶导数，梯度'''
        return y_pred - y_true # 负的残差

    def h(self,y_true,y_pred):
        '''二阶导数'''
        return np.ones_like(y_true)  # 二阶是导数是1

class LogisticLoss:
    '''
    逻辑损失,用于二分类，此时与 Y 的label 取值为{1,-1},用-1 取代 0 标签
    '''
    def g(self,y_true,y_pred):
        '''一阶导数'''
        return 1 - y_true - 1 / (1 + np.exp(y_pred))

    def h(self,y_true,y_pred):
        '''二阶导数'''
        return np.exp(y_pred) / np.power((1 + np.exp(y_pred)),2)

class CrossEntropyLoss:
    '''交叉熵损失，用于多分类，此时 Y 的label 为one-hot编码'''
    def g(self,y_true,y_pred):
        '''一阶导数'''
        # 一阶导数实际为(y_pred - y_true),但为防止其过大导致np.exp()溢出，在损失函数的基础上除以N，N为样本数量
        return (y_pred - y_true) / len(y_pred)

    def h(self,y_true,y_pred):
        '''二阶导数'''
        return np.ones_like(y_true.shape) / len(y_true)