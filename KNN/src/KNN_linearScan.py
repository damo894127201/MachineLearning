# -*- coding: utf-8 -*-
# @Time    : 2019/9/22 12:43
# @Author  : Weiyang
# @File    : KNN_linearScan.py

#=============================================================================================
# KNN:线性扫描实现KNN
# 计算样本与所有训练实例的欧式距离，从中取出距离最小的前K个样本；
# 分类时：由这K个样本的多数类作为预测样本的类别；
# 回归时：由这K个样本的均值作为预测样本的y值
#=============================================================================================


import numpy as np
import math

class KNN_linearScan:
    '''线性扫描实现KNN'''

    def __init__(self,K=5):
        self.K = K # K个最近邻

    def euclidean_distance(self,x1, x2):
        """ Calculates the l2 distance between two vectors """
        distance = 0
        # Squared distance between each coordinate
        for i in range(len(x1)):
            distance += pow((x1[i] - x2[i]), 2)
        return math.sqrt(distance)

    def normalize(self,X, axis=-1, order=2):
        """ Normalize the dataset X """
        l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
        l2[l2 == 0] = 1 # 防止模长为0
        return X / np.expand_dims(l2, axis)

    def predict(self, X_test, X_train, y_train,isRegression=False):
        '''由于KNN没有显示地学习过程，因此直接预测'''

        # 对训练数据和预测数据归一化，消除量纲影响
        X_train = self.normalize(X_train)
        X_test = self.normalize(X_test)

        y_predict = np.zeros(X_test.shape[0]) # 存放预测结果
        data = [] # 用于存放最近邻的k个训练实例
        for i in range(X_test.shape[0]):
            # 测试的数据和训练的各个数据的欧式距离,以及相应的label或y值
            distances = np.zeros((X_train.shape[0], 2))
            for j in range(X_train.shape[0]):
                dis = self.euclidean_distance(X_test[i], X_train[j]) #计算欧式距离
                label = y_train[j] #测试集中每个训练数据的分类标签或y值
                distances[j] = [dis, label]
            # argsort()得到测试集到各个训练数据的欧式距离，从小到大排列并且得到序列，然后再取前k个.
            k_nearest_neighbors = distances[distances[:,0].argsort()][:self.K]
            # 得到K个距离最近的训练实例
            k_data = X_train[distances[:,0].argsort()][:self.K]
            data.append(k_data)

            # 判断是否是回归
            if isRegression:
                y_pred = np.array(k_nearest_neighbors[:,1]).mean()
                y_predict[i] = y_pred
            else:
                # 分类
                # 利用np.bincount统计k个近邻里面各类别出现的次数，并返回类别对应的索引
                counts = np.bincount(k_nearest_neighbors[:, 1].astype('int'))
                # 得出每个测试数据k个近邻里面各类别出现的次数最多的类别
                testLabel = counts.argmax()
                y_predict[i] = testLabel
        return y_predict,data

if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    # 计算准确率
    def accuracy_score(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
        return accuracy

    print('--------KNN Linear Scan Classification  --------')
    data = datasets.load_iris()
    X = data.data
    Y = data.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

    model = KNN_linearScan(K=5)
    y_pred,_ = model.predict(X_test,X_train,Y_train,isRegression=False)
    accuracy = accuracy_score(Y_test, y_pred)
    print("Accuracy:", accuracy)