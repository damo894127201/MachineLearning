# -*- coding: utf-8 -*-
# @Time    : 2019/9/15 18:46
# @Author  : Weiyang
# @File    : GBDT.py

#=====================================================================================================
# GBDT梯度提升决策树模型(Gradient Boosting Decision Tree)
# 该模型可用于分类和回归
# GBDT = 提升树 + 梯度提升 = (提升方法 + 决策树) + 梯度提升
#      = (加法模型 + 前向分步算法 + 决策树) + 梯度提升(用损失函数的负梯度来近似残差)
# 提升树是以分类或回归树为基本分类器的提升方法，显然提升树是一种集成思想
# 提升树使用残差(=真实值-预测值)来学习,GBDT用损失函数的负梯度来近似残差(通过求损失函数对回归树的偏导可得到)
#=====================================================================================================

from RegressionDecisionTree import RegressionDecisionTree
import numpy as np
from LossFunction import *
from tqdm import tqdm

class GBDT:
    '''
    GBDT类，该类是GBDT分类树和GBDT回归树的父类
    GBDT使用的基分类器是回归树!!! GBDT使用一组回归树来训练预测损失函数的梯度
    '''
    def __init__(self,n_estimators,learning_rate,min_samples_split,
                 min_impurity,max_depth,isRegression):
        self.n_estimators = n_estimators # 用作基分类器的回归树的数量
        self.learning_rate = learning_rate # 梯度下降的学习率
        self.min_samples_split = min_samples_split # 每棵子树的节点最小数目，小于后不再继续切割(用于构造回归树)
        self.min_impurity = min_impurity # 每棵子树的最小纯度，小于后不再继续切割(用于构造回归树)
        self.max_depth = max_depth # 每棵回归树的最大深度，大于后不再继续切割节点(用于构造回归树)
        self.isRegression = isRegression # 为True时为回归问题，False为分类问题
        self.loss = SquareLoss() # 平方损失函数
        # 如果是用GBDT分类，则损失函数为
        if not self.isRegression:
            self.loss = SoftmaxLoss()

        # GBDT无论是用来做分类问题，还是回归问题，其基分类器都是回归树
        # 如果是分类问题，则也是利用残差(近似为负梯度)来学习输出类别的概率
        self.trees = [] # 存储基分类器
        for tree_index in range(self.n_estimators):
            self.trees.append(RegressionDecisionTree(min_samples_split=self.min_samples_split,
                                                     min_impurity=self.min_impurity,
                                                     max_depth=self.max_depth))

    def fit(self,X,Y):
        '''训练GBDT'''
        # 第一棵树开始拟合数据
        self.trees[0].fit(X,Y)
        # 第一棵树的预测值
        y_pred = self.trees[0].predict(X)
        if len(Y.shape) != len(np.array(y_pred).shape):
            Y = np.expand_dims(Y,axis=1)
        # 展示训练进度
        for i in tqdm(range(1,self.n_estimators)):
            # 残差 = 负梯度
            rmi = self.loss.NegativeGradient(Y,y_pred)
            # 第i轮，对负梯度拟合一个回归树
            self.trees[i].fit(X,rmi)
            # 第i棵树的预测值
            y_pred = self.trees[i].predict(X)
            # 这里我们给y_pred设置一个学习率，再减去右边项后，它会使得下一步要学习的残差变小
            # 同理，为保持一致性，再预测时，我们还要将其再加回来
            y_pred -= np.multiply(self.learning_rate,y_pred)

    def predict(self,X):
        '''模型预测'''
        # 预测值
        y_pred = self.trees[0].predict(X)
        for i in range(1,self.n_estimators):
            # 将每个模型的预测结果累加起来
            y_pred += np.multiply(self.learning_rate,self.trees[i].predict(X))
            #y_pred += np.array(self.trees[i].predict(X))
        # 如果是分类问题，此时Y是经过one-hot编码的label,因此y_pred不在是一个标量，而是一个多维的数组
        # 假设类别数为3，则
        # Y = np.array([[1,0,0],[0,1,0],...]),y_pred = np.array([[value1,value2,value3],...]])
        # 则需将y_pred转为概率,用softmax转概率
        if not self.isRegression:
            # 转为概率分布
            y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred),axis=1),axis=1)
            # 将概率转为label，即取维度值最大的y_pred所对应的索引
            y_pred = np.argmax(y_pred,axis=1)
            return y_pred
        return np.squeeze(y_pred) # 回归,返回预测值时去除多余的维度