# -*- coding: utf-8 -*-
# @Time    : 2019/9/17 19:09
# @Author  : Weiyang
# @File    : XGBoost.py

#============================================
# XGBoost模型：
# XGBoost算法的前向分步算法实现
# 该类是XGBoost用于分类和回归的基类
#============================================

from XGBoostRegressionDecisionTree import XGBoostRegressionDecisionTree
from LossFunction import LeastSquareLoss,LogisticLoss,CrossEntropyLoss
import numpy as np
from tqdm import tqdm


class XGBoost:
    '''该类是XGBoost用于分类和回归的基类'''
    def __init__(self,n_estimators=200,min_samples_split=2,
                 min_impurity=1e-7,max_depth=float("inf"),isRegression=True,lambd=0.1,gama=0.1,
                 objective="logistic"):
        self.n_estimators = n_estimators # 基模型的数量，即树的个数
        self.min_samples_split = min_samples_split # 拆分节点时所需要的最小样本数
        self.min_impurity = min_impurity # 分裂节点时最小的损失函数减少值
        self.max_depth = max_depth # 树的最大深度
        self.isRegression = isRegression # XGBoost是否用于回归
        self.lambd = lambd  # lambd 是损失函数中,正则化项 (所有叶节点值平方和sum(w^2)) 的系数
        self.gama = gama  # gama 是损失函数中，正则化项 叶节点个数|T| 的系数
        self.objective = objective # 判断是是多分类,还是二分类 objective = 'logistic' or 'crossEntropy'
        # 基分类器的损失函数
        if self.isRegression : # 均方损失函数，用于回归
            self.loss = LeastSquareLoss()
        else:
            # 分类问题，判断是否二分类还是多分类
            if self.objective == "logistic":
                self.loss = LogisticLoss() # 逻辑损失，用于二分类
            elif self.objective == "crossEntropy": # 交叉熵损失，用于多分类
                self.loss = CrossEntropyLoss()
        # 初始化决策树
        self.trees = []
        for _ in range(n_estimators):
            tree = XGBoostRegressionDecisionTree(min_samples_split=self.min_samples_split,
                                                 min_impurity=self.min_impurity,
                                                 max_depth=self.max_depth,
                                                 loss=self.loss,
                                                 lambd=self.lambd,
                                                 gama=self.gama)
            self.trees.append(tree)

    def fit(self,X,Y,num_label=None):
        '''
        训练模型
        X = np.array([[value,value,...],...])
        分类时：Y = [label,label,...] ；回归时，Y = [value,value,...]
        如果是分类问题，num_label表示类别数
        '''
        # 对Y 预处理
        # 如果是分类问题，便需要将label转为one-hot编码
        if not self.isRegression:
            # 我们首先需要对Y进行one-hot编码
            if not num_label:
                num_label = np.amax(Y) + 1  # 获取类别的最大值+1即为类别数,从0开始计数
            # 如果是多分类
            if self.objective == 'crossEntropy':
                one_hot = np.zeros((len(Y), num_label))
                for i in range(len(Y)):
                    one_hot[i, Y[i]] = 1
                Y = one_hot
            elif self.objective == 'logistic':
                # 将label为0的标签转为-1，label为1的标签仍为1
                temp = []
                for label in Y:
                    if label == 0:
                        temp.append(-1)
                    else:
                        temp.append(1)
                Y = np.expand_dims(temp,axis=1)
        else:
            # 回归时扩充一个维度
            Y = np.expand_dims(Y,axis=1)
        Y_pred = np.zeros(Y.shape) # 用于存储预测值,第一轮开始前预测值为0
        # 前向分步训练
        for i in tqdm(range(self.n_estimators)):
            tree = self.trees[i] # 当前树
            # 合并 Y 和 Y_pred ,这是XGBoost训练时所需要的
            Y_and_pred = np.concatenate((Y,Y_pred),axis=1)
            # 训练回归树
            tree.fit(X,Y_and_pred)
            update_pred = tree.predict(X) # 当前树的预测值
            update_pred = np.reshape(update_pred,(X.shape[0],-1))
            # 加上当前树预测的值到Y_pred
            Y_pred = Y_pred + update_pred

    def predict(self,X):
        '''模型预测'''
        Y_pred = None # 存储预测值
        # 开始预测
        for tree in self.trees:
            update_pred = tree.predict(X) # 当前树的预测值
            update_pred = np.reshape(update_pred,(X.shape[0],-1))
            if Y_pred is None:
                Y_pred = np.zeros_like(update_pred)
            Y_pred = Y_pred + update_pred # 将当前树的预测值累加到Y_pred中
        # 如果是分类问题，此时Y是经过one-hot编码的label,因此y_pred不在是一个标量，而是一个多维的数组
        # 假设类别数为3，则
        # Y = np.array([[1,0,0],[0,1,0],...]),Y_pred = np.array([[value1,value2,value3],...]])
        # 则需将Y_pred转为概率,用softmax转概率
        # 如果是分类
        if not self.isRegression:
            # 多分类
            if self.objective == "crossEntropy":
                # 转为概率分布
                Y_pred = np.exp(Y_pred) / np.expand_dims(np.sum(np.exp(Y_pred), axis=1), axis=1)
                # 将概率转为label，即取维度值最大的y_pred所对应的索引
                Y_pred = np.argmax(Y_pred, axis=1)
                return Y_pred
            # 二分类
            # 由于二分类中模型用的标签实际是1和-1，因此我们将预测值大于0的设置为正类，小于0的设置为负类
            elif self.objective == "logistic":
                temp = []
                for value in Y_pred:
                    # 值大于0的，类别预测为1，小于等于0的，类别预测为0
                    if value > 0:
                        temp.append(1)
                    else:
                        temp.append(0)
                return np.array(temp)
        return np.squeeze(Y_pred)  # 回归,返回预测值时去除多余的维度