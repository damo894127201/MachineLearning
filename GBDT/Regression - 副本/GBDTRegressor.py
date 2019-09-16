# -*- coding: utf-8 -*-
# @Time    : 2019/9/15 21:35
# @Author  : Weiyang
# @File    : GBDTRegressor.py

#==================================
# GBDT用于 回归的类
#==================================

from GBDT import GBDT
import numpy as np

class GBDTRegressor(GBDT):
    '''GBDT回归器'''
    def __init__(self,n_estimators=200,learning_rate=0.5,min_samples_split=2,
                 min_var_red=1e-7,max_depth=4):
        super(GBDTRegressor,self).__init__(n_estimators=n_estimators,
                                           learning_rate=learning_rate,
                                           min_samples_split=min_samples_split,
                                           min_impurity=min_var_red,# 最小的平方误差
                                           max_depth=max_depth,
                                           isRegression=True)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    def mean_squared_error(y_true, y_pred):
        """ Returns the mean squared error between y_true and y_pred """
        mse = np.mean(np.power(y_true - y_pred, 2))
        return mse

    # 生成数据
    def gen_data(x1, x2):
        y = np.sin(x1) * 1 / 2 + np.cos(x2) * 1 / 2 + 0.1 * x1
        return y

    def load_data():
        x1_train = np.linspace(0, 50, 500)
        x2_train = np.linspace(-10, 10, 500)
        data_train = np.array(
            [[x1, x2, gen_data(x1, x2) + np.random.random(1) - 0.5] for x1, x2 in zip(x1_train, x2_train)])
        x1_test = np.linspace(0, 50, 100) + np.random.random(100) * 0.5
        x2_test = np.linspace(-10, 10, 100) + 0.02 * np.random.random(100)
        data_test = np.array([[x1, x2,gen_data(x1, x2)] for x1, x2 in zip(x1_test, x2_test)])
        return data_train, data_test

    print('-------- Regression Tree --------')
    '''
    train, test = load_data()
    # train的前两列是x，后一列是y，这里的y有随机噪声
    X_train, Y_train = train[:, :2], train[:, 2]
    X_test, Y_test = test[:, :2], test[:, 2]  # 同上，但这里的y没有噪声
    

    # 我们的模型
    clf = GBDTRegressor()
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    '''

    # Load temperature data
    import pandas as pd
    data = pd.read_csv('./TempLinkoping2016.txt', sep="\t")

    time = np.atleast_2d(data["time"].as_matrix()).T
    temp = np.atleast_2d(data["temp"].as_matrix()).T

    X = time.reshape((-1, 1))  # Time. Fraction of the year [0, 1]
    X = np.insert(X, 0, values=1, axis=1)  # Insert bias term
    y = temp[:, 0]  # Temperature. Reduce to one-dim

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.5)

    model = GBDTRegressor()
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(Y_test, y_pred)

    print("Mean Squared Error:", mse)

    # sklearn的模型
    from sklearn import tree

    clf = tree.DecisionTreeRegressor()
    clf.fit(X_train, Y_train)
    y_pred_sk = clf.predict(X_test)

    plt.figure()
    plt.plot(np.arange(len(Y_train)), Y_test, "go-", label="True value")
    plt.plot(np.arange(len(Y_train)), y_pred, "ro-", label="Predict value")
    plt.plot(np.arange(len(Y_train)), y_pred_sk, "bo-", label="Sklearn value")
    plt.title("True value && Predict value && Sklearn value")
    plt.legend(loc="best")
    plt.show()