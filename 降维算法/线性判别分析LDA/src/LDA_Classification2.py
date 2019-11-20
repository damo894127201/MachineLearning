# -*- coding: utf-8 -*-
# @Time    : 2019/11/18 13:39
# @Author  : Weiyang
# @File    : LDA_Classification2.py

#======================================================================================================================
# LDA分类算法：
# 1. 算法原理
#   LDA分类基本思想是假设各个类别的样本数据符合高斯分布，各个特征之间相互独立，这样利用LDA进行投影后，
#   可以利用极大似然估计 计算各个类别投影数据的均值和方差，进而得到该类别高斯分布的概率密度函数。
# 2. 预测原理
#   当一个新的样本到来后，我们可以将它投影，然后将投影后的样本特征分别带入各个类别的高斯分布概率密度函数，
#   计算它属于这个类别的概率，最大的概率对应的类别即为预测类别。
# 3. 参数估计：极大似然估计，写出似然函数(由于各个特征之间相互独立，可以用各个特征的概率密度函数的乘积取对数得到，之后对参数
#              求导即可得到，各个特征的均值和方差等于数据中各自特征的样本均值和样本方差)
# 4. 参数：与LDA_Classification.py的区别是，我们对每一个特征应用一次一维高斯分布，即求出每一个特征的均值和方差，最后将各个特征的
#          P(x|y)连乘得到概率(取对数方便处理)
#======================================================================================================================

from LDA import LDA
import numpy as np
import math
from collections import defaultdict


class LDA_Classification(object):
    '''LDA分类算法'''
    def __init__(self,k):
        self.k = k # 降维后的数据的维度
        self.W = None # 投影矩阵
        self.classes = None # 存储类别
        self.parameters = {} # 存储各个类别的数据的均值向量、协方差矩阵的逆矩阵以及协方差矩阵的行列式值，
                             # key是class+label,value是{'mean':mean,'var':var}

    def fit(self,data):
        '''训练模型参数,data = np.array([[value1,value2,...,label],...])'''
        model = LDA(k=self.k)
        new_data,self.W = model.fit(data) # new_data是每行一条数据，最后一维度是类别
        new_data = np.array(new_data) # matrix -> np.ndarray
        print('降维后的前3条数据为',new_data[:3])
        # 按类别将降维后的数据分类
        m,n = new_data.shape # m条数据,每条数据由n个维度，其中最后一维度是类别
        # 将数据按类别划分
        dataDict = defaultdict(list)
        # 遍历数据
        for i in range(m):
            dataDict[new_data[i,-1]].append(new_data[i,:-1]) # 去除类别标签
        self.classes = list(dataDict.keys()) # 存储类别
        # 计算每个类别的均值向量，协方差矩阵和协方差矩阵的行列式的值
        for label in dataDict.keys():
            temp = np.array(dataDict[label]) # 每行为一条数据
            mean = temp.mean(axis=0,keepdims=True) # 求均值向量，保持二维
            var = temp.var(axis=0,keepdims=True)
            self.parameters['class' + str(label)] = {'mean':mean,'var':var}

    def _pdf(self,X,label):
        '''
        计算在y=c的条件下，P(X|y=c)=P(x1|y=c)P(x2|y=c)...P(xn|y=c)的值
        X = array([[x1,x2,..,xn],[x1,...,xn],..]),label是一个整数，表示类别
        '''
        eps = 1e-4 # 防止分母为0的一个正数
        # shape = [1,n_features]
        mean = self.parameters["class" + str(label)]["mean"] # 获取类别y=c时，各个特征的均值
        var = self.parameters["class" + str(label)]["var"] # 获取类别y=c时，各个特征的方差

        # 用高斯模型来估计概率
        # 一维高斯分布的概率密度函数
        # numerator.shape = [m_sample,feature]
        numerator = np.exp(-(X - mean)**2 / (2 * var + eps)) # 概率密度函数的分子，eps防止分母为0
        denominator = np.sqrt(2 * np.pi * var + eps)# 概率密度函数的分母

        # 每个特征之间条件独立
        # p(x1,x2,x3|y)=p(x1|y)*p(x2|y)*p(x3|y),取对数后，相乘变为相加，shape=(n_samples,1)
        probability = np.sum(np.log(numerator/denominator + eps),axis=1,keepdims=True) # 对数运算防止数值溢出
        probability = probability.reshape(1,-1) # shape=(1,n_samples)
        probability = probability[0] # (n_samples,)
        return probability

    def _predict(self,X):
        '''
        计算每个类别的概率P(y|x1,x2,x3)，不考虑分母的情况下
        P(y|x1,x2,x3) = P(y)*P(x1|y)*P(x2|y)*P(x3|y)
        '''
        probas = [] # 存储概率
        # 遍历每个类别
        for label in self.classes:
            posterior = self._pdf(X,label) # 后验概率，由于取了对数，因此相乘便相加,shape=[n_samples,n_label]
            probas.append(posterior)
        probas = np.array(probas)
        return probas # shape=[n_label,n_samples]

    def predict(self,X):
        '''取概率最大的类别作为预测值,X = np.array([[value,..],...],'''
        if len(X.shape) < 2:
            X = np.array(X)
        # 对X进行LDA降维
        new_X = self.W.T.dot(X.T)
        new_X = np.array(new_X).T # 每行一条数据
        probas = self._predict(new_X) # shape=[n_label,n_samples]
        prediction = np.argmax(probas,axis=0) # shape = [1,n_samples]
        return prediction


if __name__ == '__main__':
    # 加载sklearn自带的手写体数据集
    from sklearn.datasets.samples_generator import make_classification
    from sklearn.model_selection import train_test_split

    def normalize(dataset,axis=-1,order=2):
        '''将数据集各个维度归一化'''
        # 求数据集最后一个维度的模或L2范数
        l2 = np.atleast_1d(np.linalg.norm(dataset,order,axis))
        l2[l2 == 0] = 1 # 如果某个维度的取值全是0，则赋值为1，避免下面分母为0
        return dataset / np.expand_dims(l2,axis)

    def accuracy_score(y_true,y_pred):
        '''计算准确率'''
        accuracy = np.sum(y_true == y_pred,axis=0) / len(y_true)
        return accuracy

    X,y = make_classification(n_samples=1000,n_features=3,n_redundant=0,n_classes=2,n_informative=2,
                              n_clusters_per_class=1,class_sep=0.5,random_state=0)

    # 分割数据集为训练集和测试集
    X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2)
    print("X_train.shape:",X_train.shape,' ',"X_test.shape:",X_test.shape)
    # 训练模型
    model = LDA_Classification(k=2)
    # 将特征和类别拼接一起，类别在最后一个维度上
    data = np.concatenate((X_train, Y_train[:, None]), axis=1)
    model.fit(data)
    # 预测模型
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test,y_pred)
    print('真实标签为：')
    print(Y_test)
    print('预测标签为：')
    print(y_pred)
    print("Accuracy: ",accuracy)