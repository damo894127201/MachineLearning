# -*- coding: utf-8 -*-
# @Time    : 2019/11/18 13:39
# @Author  : Weiyang
# @File    : LDA_Classification.py

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
# 4. 参数：均值向量，协方差矩阵，协方差矩阵对应的行列式值
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
                             # key是class+label,value是{'mean':mean,'inverse_cov':inverse_cov,'determinant':determinant}

    def fit(self,data):
        '''训练模型参数,data = np.array([[value1,value2,...,label],...])'''
        model = LDA(k=self.k)
        new_data,self.W = model.fit(data) # new_data是每行一条数据，最后一维度是类别
        new_data = np.array(new_data) # matrix -> np.ndarray
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
            temp = np.array(dataDict[label]).T # 转为每列为一条数据
            mean = temp.mean(axis=1,keepdims=True) # 求均值向量，保持二维
            covariance = (temp - mean).dot((temp - mean).T) # 求协方差矩阵,shape=(n,n),n是特征的个数
            determinant = np.linalg.det(covariance) # 求协方差矩阵的行列式
            inverse_cov = np.mat(covariance).I # 协方差矩阵的逆矩阵
            self.parameters['class' + str(label)] = {'mean':mean,'inverse_cov':inverse_cov,'determinant':determinant}

    def _pdf(self,X,label):
        '''
        计算样本是某个类别的概率
        计算在y=c的条件下，P(X|y=c)的值
        X = array([[x1,x2,..,xn],[x1,...,xn],..])，即多条数据，不带类别，每行一条数据 shape = (n_samples,n_features)
        label表示类别
        '''
        eps = 1e-4 # 防止分母为0的一个正数
        mean = self.parameters["class" + str(label)]["mean"] # 获取类别y=c时，特征均值向量 shape = [n_features,1]
        inverse_cov = self.parameters["class" + str(label)]["inverse_cov"] # 获取类别y=c时，协方差矩阵
                                                                           # shape = [n_features,n_features]
        determinant = self.parameters["class" + str(label)]["determinant"] # 获取类别y=c时，协方差矩阵的行列式,一个标量

        # 用多维高斯模型来估计概率，特征之间相互独立
        # 将X转置为每列为一条数据
        new_X = X.T # shape=(n_features,n_samples)
        # numerator.shape = [m_samples,m_samples]
        # 整体计算数据集
        numerator = np.exp(-1/2 * ((new_X - mean).T.dot(inverse_cov.dot(new_X - mean)))) # 概率密度函数的分子
        denominator = np.sqrt(math.pow(2 * np.pi,new_X.shape[0]) * determinant + eps)# 概率密度函数的分母

        # shape = [1,n_samples]
        probability = np.log(numerator/denominator + eps) # 对概率取对数运算，防止数值溢出
        probability = np.diag(probability) # 真正有用的是对角线上的元素，分别对应每条数据属于当前类别的概率
                                           # shape=(n_samples,)
        '''
        # 遍历每条数据
        probability = []
        denominator = np.sqrt(math.pow(2 * np.pi, new_X.shape[0]) * determinant + eps)  # 概率密度函数的分母
        # 遍历每条数据
        for i in range(new_X.shape[1]):
            # 概率密度函数的分子
            numerator = np.exp(-1 / 2 * ((new_X[:,i][:,None] - mean).T.dot(inverse_cov.dot(new_X[:,i][:,None] - mean))))
            proba = np.log(numerator / denominator + eps)  # 对概率取对数运算，防止数值溢出
            probability.append(proba)
        '''
        return probability

    def _predict(self,X):
        '''
        计算每个类别的概率P(y|x1,x2,x3)
        '''
        probas = [] # 存储概率
        # 遍历每个类别
        for label in self.classes:
            posterior = self._pdf(X,label) # 后验概率 shape=[1,n_samples]
            probas.append(posterior)
        return probas # shape=[n_label,n_samples]

    def predict(self,X):
        '''取概率最大的类别作为预测值,X = np.array([[value,..],...],'''
        if len(X.shape) < 2:
            X = np.array(X)
        # 对X进行降维
        new_X = self.W.T.dot(X.T)
        new_X = np.array(new_X).T # 每行一条数据
        probas = self._predict(new_X) # [matrix,matrix,...]
        probas = np.array(probas) #matrxi->np.ndarray
        probas = probas.reshape(len(self.classes),-1) # shape=[n_label,n_samples]
        prediction = np.argmax(probas,axis=0) # shape = [1,n_samples]
        return prediction

if __name__ == '__main__':
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