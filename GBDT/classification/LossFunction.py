# -*- coding: utf-8 -*-
# @Time    : 2019/9/15 19:31
# @Author  : Weiyang
# @File    : LossFunction.py

#==========================================
# 平方损失函数,交叉熵损失函数和softmax损失函数
#==========================================

import numpy as np

class Loss:
    """损失函数的基类"""
    def loss(self,y_true,y_pred):
        return NotImplementedError()

    def NegativeGradient(self,y,y_pred):
        raise NotImplementedError()

    def accuracy(self,y,y_pred):
        return 0

class SquareLoss(Loss):
    '''平方损失函数'''
    def __init__(self):pass

    def loss(self,y,y_pred):
        """返回平方损失"""
        return 0.5 * np.power((y - y_pred),2)

    def NegativeGradient(self,y,y_pred):
        """返回负梯度"""
        return -(y - y_pred) # (y - y_pred) 是残差

class CrossEntropy(Loss):
    '''
    交叉熵损失函数= - sum(y_i * logP(x_i)) ，y_i是真实标签，P(x_i)预测的概率
    可用于二分类和多分类，用于多分类时，要把标签转为one-hot编码,这样y={1,0}
    '''
    def __init__(self):pass

    def loss(self,y,p):
        '''y是真实的标签,p是预测的概率'''
        # 避免除0
        # 这个函数将p中的数据限制在[1e-15,1-1e-15],大于1-1e-15的赋值为1-1e-15
        # 小于1e-15，赋值为1e-15
        p = np.clip(p,1e-15,1-1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def accuracy(self,y,p):
        '''y = np.array([[1,0,0,..],...]),y = np.array([[1,0,0,..],...])'''
        y_label = np.argmax(y,axis=1)
        p_label = np.argmax(p,axis=1)
        return np.sum(y_label == p_label,axis=0) / len(y_label)

    def NegativeGradient(self,y,p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (- (y / p) + (1 - y) / (1 - p))

class SoftmaxLoss(Loss):
    '''softmax损失函数'''
    def loss(self,y,p):
        '''softmax损失函数实际上是交叉熵损失函数'''
        L = CrossEntropy()
        # 将p转为概率
        p = np.exp(p) / np.expand_dims(np.sum(np.exp(p), axis=1), axis=1)
        return L.loss(y,p)

    def NegativeGradient(self,y,p):
        return -(p - y) # 梯度为 (p - y)