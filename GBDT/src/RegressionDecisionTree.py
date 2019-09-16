# -*- coding: utf-8 -*-
# @Time    : 2019/9/13 14:44
# @Author  : Weiyang
# @File    : RegressionDecisionTree.py

#######################################################################
# CART回归决策树模型
# 决策树模型是一种分类与回归方法，可以视为if-then规则的集合，也可以认为是定义在
#             特征空间与类空间上的条件概率分布。

# 决策树学习包含三个部分：特征选择，决策树生成和决策树剪枝
# 特征选择的指标：平方误差最小
# 决策树生成算法：CART算法
# 决策树剪枝：防止过拟合，目标是使决策树损失函数最小
#######################################################################

from DecisionTree import DecisionTree
import numpy as np

class RegressionDecisionTree(DecisionTree):
    '''CART回归决策树模型'''
    def calculate_variance(self,Y):
        '''计算平方误差sum((y-y_pred)^2)/N=Var(y),Y = np.array([value1,value2,...])'''
        import numpy as np
        Y = np.squeeze(Y)
        return np.sum(np.var(Y))

    def calculate_variance_reduction(self,y,y1,y2):
        '''
        计算在当前特征的划分下，各个划分单元的平方误差,越小越好
        y:完整的标签集合，y1和y2是y，基于某个特征的阈值拆分成两部分的标签集合
        '''
        y = np.squeeze(y)
        var_y = self.calculate_variance(y)
        var_y1 = self.calculate_variance(y1)
        var_y2 = self.calculate_variance(y2)
        # 平方误差减少的量
        p1 = float(len(y1)) / len(y)
        p2 = float(len(y2)) / len(y)
        reduction = var_y - p1 * var_y1 - p2 * var_y2
        #variance = p1 * var_y1 + p2 * var_y2
        return reduction

    def mean_of_y(self,y):
        '''当前节点输出的预测值为当前节点内所有样本y值的均值'''
        y = np.squeeze(y)
        return y.mean()

    def fit(self,X,Y):
        '''
        训练模型
        X = np.array([[value,value,...],...]),Y = [label1,label2,...]
        '''
        self.impurity_calculation = self.calculate_variance_reduction
        self.leaf_value_calculation = self.mean_of_y
        super(RegressionDecisionTree,self).fit(X,Y)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
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
    train, test = load_data()
    # train的前两列是x，后一列是y，这里的y有随机噪声
    X_train, Y_train = train[:, :2], train[:, 2]
    X_test, Y_test = test[:, :2], test[:, 2]  # 同上，但这里的y没有噪声

    # 我们的模型
    clf = RegressionDecisionTree()
    clf.fit(X_train,Y_train)
    #clf.print_tree()
    y_pred = clf.predict(X_test)

    # sklearn的模型
    from sklearn import tree
    clf = tree.DecisionTreeRegressor()
    clf.fit(X_train,Y_train)
    y_pred_sk = clf.predict(X_test)

    plt.figure()
    plt.plot(np.arange(len(y_pred)),Y_test,"go-",label="True value")
    plt.plot(np.arange(len(y_pred)),y_pred,"ro-",label="Predict value")
    plt.plot(np.arange(len(y_pred)), y_pred_sk, "bo-", label="Sklearn value")
    plt.title("True value && Predict value && Sklearn value")
    plt.legend(loc="best")
    plt.show()

