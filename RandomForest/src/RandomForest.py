# -*- coding: utf-8 -*-
# @Time    : 2019/9/17 9:44
# @Author  : Weiyang
# @File    : RandomForest.py

#============================================================================================
# 随机森林分类器模型：分类模型，集体决策，多数表决
# 随机森林分类器使用一系列分类树作为基分类器，在数据集的子集上和特征集的随机子集上进行训练。
# 随机性体现在：每棵树训练样本的随机和每棵树的分裂特征集合也是随机的

# 随机森林算法属于集成学习(两大分支：Boosting和Bagging)的一种，基于Bagging思想实现。
# Bagging思想与Boosting串行训练方式不同，Bagging方法在训练过程中，各基分类器之间无强依赖，
# 可以进行并行训练，比较著名的便是随机森林算法。Bagging为了让基分类器之间互相独立，将训练集分为若干
# 个子集，各个子集之间的样本可以存着重叠，分割子集的方法基于Boostrap,即有放回的抽样，
# 因此大约有36.8%的数据不会被抽到。

# Bagging思想是利用集体投票表决，即将多个基分类器的结果汇总，然后采取多数投票的方式表决。

# 随机森林算法基于决策树基分类器，主要有以下点：
# 1. 基分类器是分类决策树，本算法采用的是ID3或C4.5决策树
# 2. 每个基分类器学习的数据集是全样本数据集的子集，子集可以重叠(有放回的抽样)
# 3. 每个基分类器学习的子集的特征数也是基于有放回随机抽样，随机特征，特征可以重复
# 4. 在预测时，我们要根据每个基分类器所利用的特征，进行预测，即不是利用全部特征让基分类器做预测
#===========================================================================================

import numpy as np
from ClassDecisionTree import ClassDecisionTree

class RandomForest:
    '''随机森林算法：分类算法'''
    def __init__(self,n_estimators=100,min_samples_split=2,min_gain=0,
                 max_depth=float("inf"),max_features=None):
        self.n_estimators = n_estimators # 参与决策的基分类器的个数
        self.min_samples_split = min_samples_split # 分裂节点时所需要的最少样本，用于预剪枝
        self.min_gain = min_gain # 分裂节点时所需要的最少信息增益或信息增益比，用于预剪枝
        self.max_depth = max_depth # 每棵树的最大深度，用于预剪枝
        self.max_features = max_features # 每棵树选用数据集中的最大特征数量
        self.tree_features = [] # 存储每棵树训练的特征子集

        # 存储每个基分类器
        self.trees = []
        # 建立随机森林
        for _ in range(self.n_estimators):
            tree = ClassDecisionTree(min_samples_split=self.min_samples_split,
                                     min_impurity=self.min_gain,
                                     max_depth=self.max_depth)
            self.trees.append(tree)

    def get_bootstrap_data(self,X,Y):
        '''
        bootstrap(自助抽样)：有放回的抽样
        X = np.array([[value,...],...])
        Y = [label,label,...]
        '''
        # 通过bootstrap的方式获得每个基分类器的数据
        n_samples = X.shape[0] # 样本数
        Y = np.array(Y).reshape(n_samples,1) # 将Y转为numpy.array，且维度长度与X一致，便于后面合并

        # 合并X和Y，方便boostrap
        X_Y = np.hstack((X,Y)) # 在水平方向上平铺
        np.random.shuffle(X_Y)

        # 数据集的子集，子集的数量与原数据一致，但数据分布并不同
        sub_datas = []
        for _ in range(self.n_estimators):
            # 参数replaced为True时代表有放回的抽样，且等概率抽样
            # 从[0,n_samples]有放回抽取n_samples个整数，有重复
            idm = np.random.choice(n_samples,n_samples,replace=True)
            sub_data = X_Y[idm,:]
            sub_X = sub_data[:,:-1] # 特征序列
            sub_Y = sub_data[:,-1] # label
            sub_datas.append([sub_X,sub_Y])
        return sub_datas

    def fit(self,X,Y,model='ID3'):
        '''
        训练随机森林
        X = np.array([[value,...],...])
        Y = [label,label,...]
        '''
        # 每棵树(基分类器)使用随机的数据集(大小与原数据相同)
        # 和随机的特征,特征可以重复，特殊数量为self.max_features
        # 获取每个基分类器的数据集
        sub_sets = self.get_bootstrap_data(X,Y)
        n_features = X.shape[1] # 原数据集特征数
        if self.max_features == None:
            self.max_features = int(np.sqrt(n_features)) # 每棵树用于分裂的特征数
        # 训练每棵树
        for i in range(self.n_estimators):
            # 获取训练子集
            sub_X,sub_Y = sub_sets[i]
            # 生成随机特征,有放回抽样，特征可以重复
            idx = np.random.choice(n_features,self.max_features,replace=True)
            # 依据随机特征，更新训练子集
            sub_X = sub_X[:,idx]
            # 训练基分类器
            self.trees[i].fit(sub_X,sub_Y,model)
            # 记录该分类器用于生成树的特征索引
            self.tree_features.append(idx)
            if i % 10 == 0:
                print('Tree ',i," train completed")

    def predict(self,X):
        '''预测结果'''
        y_preds = [] # 存储每棵树的预测结果
        for i in range(self.n_estimators):
            idx = self.tree_features[i] # 每棵树的特征子集
            sub_X = X[:,idx] # 每棵树的训练集
            y_pred = self.trees[i].predict(sub_X) # 单棵树预测结果
            y_preds.append(y_pred)
        y_preds = np.array(y_preds) # shape=(n_estimators,n_samples)
        y_preds = y_preds.T # shape = (n_samples,n_estimators)
        temp = [] # 存储随机森林集体投票的预测结果
        for y_p in y_preds:
            # np.bincount()可以统计数组中每个索引出现的次数
            temp.append(np.bincount(y_p.astype('int')).argmax())
        return temp

if __name__ == '__main__':
    # 导入sklearn自带的鸢尾花数据集
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    #计算准确率
    def accuracy_score(y_true,y_pred):
        accuracy = np.sum(y_true == y_pred,axis=0) / len(y_true)
        return accuracy
    print('-------- RandomForest Classification --------')
    data = datasets.load_iris()
    X = data.data
    Y = data.target
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
    for model in ["ID3","C4.5"]:
        clf = RandomForest()
        clf.fit(X_train,Y_train,model=model)
        y_pred = clf.predict(X_test)
        y_pred = np.array(y_pred)
        accuracy = accuracy_score(Y_test,y_pred)
        print(model + "  Accuracy:",accuracy)
        print()







