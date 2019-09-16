# -*- coding: utf-8 -*-
# @Time    : 2019/9/15 21:34
# @Author  : Weiyang
# @File    : GBDTClassifier.py

#==================================
# GBDT用于 分类的类
#==================================

from GBDT import GBDT
import numpy as np

class GBDTClassifier(GBDT):
    '''GBDT分类器'''
    def __init__(self, n_estimators=200, learning_rate=.5, min_samples_split=3,
                 min_info_gain=1e-7, max_depth=float('inf')):
        super(GBDTClassifier, self).__init__(n_estimators=n_estimators,
                                             learning_rate=learning_rate,
                                             min_samples_split=min_samples_split,
                                             min_impurity=min_info_gain,# 最小的信息增益
                                             max_depth=max_depth,
                                             isRegression=False)
    def fit(self,X,Y,num_label=None):
        '''Y = [label1,label2,...]'''
        # 我们首先需要对Y进行one-hot编码
        if not num_label:
            num_label = np.amax(Y) + 1 # 获取类别的最大值+1即为类别数,从0开始计数
        one_hot = np.zeros((len(Y),num_label))
        for i in range(len(Y)):
            one_hot[i,Y[i]] = 1
        super(GBDTClassifier,self).fit(X,one_hot)

if __name__ =='__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    # 计算准确率
    def accuracy_score(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
        return accuracy
    print('--------GBDT Classification Tree --------')
    data = datasets.load_iris()
    X = data.data
    Y = data.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)

    model = GBDTClassifier()
    model.fit(X_train,Y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test,y_pred)
    print("Accuracy:", accuracy)