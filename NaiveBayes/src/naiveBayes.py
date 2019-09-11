# -*- coding: utf-8 -*-
# @Time    : 2019/9/11 14:35
# @Author  : Weiyang
# @File    : naiveBayes.py

#############################################################################################################################
# 朴素贝叶斯分类算法
# 概念：对于给定的训练集，首先基于特征条件独立假设学习输入输出的联合概率分布P(X,Y)，
#       然后求得后验概率分布P(Y|X)=P(X,Y)/P(X)，选取使后验概率最大的那个类别。实际上
#       对于每个类别，P(X)的值都是一样的，都是P(X)=P(X1)..P(Xn)，因而求后验概率等价于求联合概率分布。
# 算法基础：贝叶斯定理和特征条件独立假设
# 算法流程：
# 1. 对于给定的训练集，计算先验概率分布P(Y)和条件概率分布P(X|Y)的估计(极大似然估计法和贝叶斯估计),这里用高斯模型来估计P(X|Y)；
# 2. 然后求得后验概率分布P(Y=Ci|X)=P(X,Y)/P(X)=P(Y=Ci)*P(X1|Y=Ci)...P(Xn|Y=Ci)/sum(P(Y=Cj)P(X1|Y=Cj)...P(Xn|Y=Cj))，对于
#    所有类别，分母的值是不变的，因此可以舍弃分母；
# 3. 第二步等价于求P(Y)P(X1|Y)...P(Xn|Y),我们的目标是计算出所有类别的后验概率分布，找出使得后验概率最大的那个类别作为输入对应的类别
# 先验概率和条件概率估计：
# 1. 极大似然估计：
# 2. 贝叶斯估计：用极大似然估计法可能出现所要估计的概率值为0的情况，即某些类别或者某些特征对应的取值没有在训练数据中出现
#                解决这一问题的方法采用贝叶斯估计。方法是在极大似然估计的基础上加上一个正数，该值取1时为拉普拉斯平滑
# 3. 本算法对条件概率估计采用高斯模型，即正态分布，求出在每个类别下各个特征的均值和方差，进而获取其概率。
#################################################################################################################################

import numpy as np

class NaiveBayes:
    '''朴素贝叶斯算法'''

    def __init__(self):
        self.classes = None # 存储类别
        self.parameters = {} # 存储模型训练后得到的参数

    def fit(self,X,Y):
        '''
        计算先验概率和条件概率
        条件概率：计算在类别y确定的条件下，X中每个特征x1,x2,...,xn的方差和平均值，
        目的在于计算条件概率P(x_i|y);
        假设类别数为3，则X和Y的格式为：
        X = array([[x1,x2,..,xn],[x1,...,xn],..]) ,Y= np.array([0,1,0,2,..])
        '''
        self.classes = np.unique(Y) # 该函数用于去除数组中的重复数字，排序后输出为数组的形式,shape=[n_classes,1]
        # 计算每个类别的先验概率，以及在类别已知的条件下各个特征的平均值和方差
        for i,c in enumerate(self.classes):
            x = X[np.where(Y == c)] # 获取属于当前类别c的所有训练数据
            prior_proba = float(x.shape[0])/X.shape[0] # 当前类别的先验概率
            # shape = [1,n_features]
            x_mean = np.mean(x,axis=0,keepdims=True) # 在当前类别c已知的条件下，每个X的特征对应的均值,保持二维特性
            x_var = np.var(x,axis=0,keepdims=True) # 在当前类别c已知的条件下，每个X的特征对应的方差，保持二维特性
            parameters = {"mean":x_mean,"var":x_var,"prior_proba":prior_proba} # 存储模型参数
            self.parameters["class" + str(c)] = parameters

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

        # 朴素贝叶斯的假设：每个特征之间条件独立
        # p(x1,x2,x3|y)=p(x1|y)*p(x2|y)*p(x3|y),取对数后，相乘变为相加
        # shape = [1,1]
        probability = np.sum(np.log(numerator/denominator + eps),axis=1,keepdims=True) # 对数运算防止数值溢出

        return probability

    def _predict(self,X):
        '''
        计算每个类别的概率P(y|x1,x2,x3)，不考虑分母的情况下
        P(y|x1,x2,x3) = P(y)*P(x1|y)*P(x2|y)*P(x3|y)
        '''
        probas = [] # 存储概率
        # 遍历每个类别
        for label in range(self.classes.shape[0]):
            prior = np.log(self.parameters["class" + str(label)]["prior_proba"]) # 类别的先验概率
            posterior = self._pdf(X,label) # 后验概率
            proba = prior + posterior # 由于取了对数，因此相乘便相加,shape=[n_samples,n_label
            probas.append(proba)
        return probas # shape=[1,1,1]

    def predict(self,X):
        '''取概率最大的类别作为预测值'''
        probas = self._predict(X) # shape=[1,1,1]
        probas = np.reshape(probas,(self.classes.shape[0],X.shape[0])) # shape=[n_classes,n_samples]
        prediction = np.argmax(probas,axis=0) # shape = [1,n_samples]
        return prediction

if __name__ == '__main__':
    # 加载sklearn自带的手写体数据集
    from sklearn import datasets
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

    digits = datasets.load_digits()
    # 将训练数据各个维度的特征值归一化，便于用高斯模型来估计概率
    X = normalize(digits.data)
    # 标签
    Y = digits.target
    # 分割数据集为训练集和测试集
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
    print("X_train.shape:",X_train.shape)
    # 训练模型
    clf = NaiveBayes()
    clf.fit(X_train,Y_train)
    # 预测模型
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(Y_test,y_pred)
    print("Accuracy: ",accuracy)