# -*- coding: utf-8 -*-
# @Time    : 2019/9/14 19:28
# @Author  : Weiyang
# @File    : LogisticRegression.py

#============================================================
# 逻辑斯蒂回归模型
# 属于分类模型
#============================================================

import numpy as np

class LogisticRegression:
    '''逻辑斯蒂回归模型'''
    def __init__(self,lr=0.1,n_iterations=4000):
        '''lr是梯度下降的学习率，n_iteration是梯度下降的轮数'''
        self.lr = lr
        self.n_iterations = n_iterations
        self.w = None # 存储训练好的模型参数
        self.b = None # 存储训练好的模型参数

    def _sigmoid(self,X):
        '''sigmoid的函数'''
        # 对sigmoid优化，避免出现极大的数据溢出
        temp = []
        for inx in X :
            if inx >= 0:
                # 始终保证np.exp(inx)的值小于1，避免极大溢出
                temp.append(1.0 / (1 + np.exp(-inx)))
            else:
                # 始终保证np.exp(inx)的值小于1，避免极大溢出
                temp.append(np.exp(inx) / (1 + np.exp(inx)))
        return np.array(temp)

    def initialize_weights(self,n_features):
        '''参数初始化'''
        # 参数范围[-1/sqrt(n_features),1/sqrt(n_features)]
        limit = np.sqrt(1 / n_features)
        w = np.random.uniform(-limit,limit,(n_features,1)) # 均匀分布初始化
        b = 0
        self.w = np.insert(w,0,b,axis=0) # 将偏置b插入到w的第一个索引位置，便于后面同时计算w和b的梯度

    def fit(self,X,Y):
        '''训练模型参数，X=np.array([[value1,value2,...],..]),Y=[label1,label2,...]'''
        # 模型样本数，特征列数
        n_samples,n_features = np.shape(X)
        # 参数初始化
        self.initialize_weights(n_features)
        X = np.insert(X,0,1,axis=1) # 给X第一列前增加一个特征列，取值全为1，用来与偏置b做乘法，得b*1=b
        Y = np.expand_dims(Y,axis=-1) # Y.shape = (n_samples,1)

        # 梯度训练n_iterations轮
        for step in range(self.n_iterations):
            # X.shape=(n_samples,n_features),w.shape=(n_features,1)
            # X.dot(self.w) = (n_samples,1)
            x_w = X.dot(self.w) # X.shape=(n_samples,n_features)
            y_pred = self._sigmoid(x_w)
            w_grad = X.T.dot(Y - y_pred) # w的梯度,w_grad.shape=(n_features,1)
            self.w = self.w + self.lr * w_grad # 更新梯度,由于是求对数极大似然的极大值，
            # 因此，我们在更新梯度时应该加上梯度，而不是减去；在求极小值时，是减去梯度，代表着往负梯度方向走误差下降最快

    def predict(self,X):
        '''模型预测，X=np.array([[value1,value2,...],..])'''
        X = np.insert(X,0,1,axis=1) # 增加一个特征列用来与b相乘
        x_w = X.dot(self.w)
        y_pred = np.round(self._sigmoid(x_w)) # 返回四舍五入的值，即>=0.5，返回1
        return y_pred.astype(int)

if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    def normalize(X,axis=-1,order=2):
        '''对数据进行归一化，X = np.array([[value1,value2,...],..])'''
        # 以X的最后一维度为准，求X中每条数据构成的向量的L2范数
        l2 = np.atleast_1d(np.linalg.norm(X,order,axis))
        l2[l2 == 0] = 1 # 对L2范数为0的，赋值为1，避免分母为0
        return X / np.expand_dims(l2,axis)

    def accuracy_score(y_true,y_pred):
        '''计算准确率'''
        accuracy = np.sum(y_true == y_pred,axis=0) / len(y_true)
        return accuracy

    # load dataset
    data = datasets.load_iris()
    X = normalize(data.data[data.target != 0]) # iris数据集有三个类别，我们需要去除label=0的类，只余2个类别
    Y = data.target[data.target != 0]
    Y[Y == 1] = 0
    Y[Y == 2] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred = np.reshape(y_pred, y_test.shape)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)