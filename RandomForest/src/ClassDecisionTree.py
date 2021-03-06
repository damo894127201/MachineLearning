# -*- coding: utf-8 -*-
# @Time    : 2019/9/13 13:51
# @Author  : Weiyang
# @File    : ClassDecisionTree.py

############################################################################
# 分类决策树模型
# 决策树模型是一种分类与回归方法，可以视为if-then规则的集合，也可以认为是定义在
#             特征空间与类空间上的条件概率分布。

# 决策树学习包含三个部分：特征选择，决策树生成和决策树剪枝
# 特征选择的指标：信息增益最大，信息增益比最大，平方误差最小和基尼指数最小
# 决策树生成算法：ID3算法，C4.5算法和CART算法
# 决策树剪枝：防止过拟合，目标是使决策树损失函数最小
############################################################################

from DecisionTree import DecisionTree
import numpy as np

class ClassDecisionTree(DecisionTree):
    '''分类决策树模型'''
    def calculate_information_gain(self,y,y1,y2):
        '''计算信息增益，值越大越好，信息增益倾向于特征取值较多的那个特征，
        y:完整的标签集合，y1和y2是y，基于某个特征的阈值拆分成两部分的标签集合'''
        y = np.squeeze(y)
        # 计算训练集y的熵
        y_entropy = self._entropy(y)
        # 计算两个子集y1和y2的熵
        y1_entropy = self._entropy(y1)
        y2_entropy = self._entropy(y2)
        # 计算在当前特征下的条件熵
        feature_entropy = float(len(y1)) / len(y) * y1_entropy + float(len(y2)) / len(y) * y2_entropy
        # 计算信息增益
        gain = y_entropy - feature_entropy
        return gain

    def calculate_information_gain_ratio(self,y,y1,y2):
        '''计算信息增益比，越大越好'''
        from math import log
        y = np.squeeze(y)
        # 信息增益
        gain = self.calculate_information_gain(y,y1,y2)
        # 训练集关于当前特征的熵
        p1 = float(len(y1)) / len(y)
        p2 = float(len(y2)) / len(y)
        feature_entropy = - p1 * log(p1,2) - p2 * log(p2,2)
        # 信息增益比
        ratio = gain / feature_entropy
        return ratio

    def majority_vote(self,y):
        '''计算叶节点中的多数类,y=np.array([label1,...])'''
        y = np.squeeze(y)
        most_common = None # 记录多数类
        max_count = 0
        for label in np.unique(y):
            # 记录当前类别的数量
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common # 返回多数类

    def fit(self,X,Y,model="ID3"):
        '''训练模型
        X = np.array([[value,value,...],...]),Y = [label1,label2,...]
        model可取"ID3","C4.5"
        '''
        if model == "ID3":
            self.impurity_calculation = self.calculate_information_gain
        elif model == "C4.5":
            self.impurity_calculation = self.calculate_information_gain_ratio

        self.leaf_value_calculation = self.majority_vote
        super(ClassDecisionTree,self).fit(X,Y)

if __name__ == "__main__":
    # 导入sklearn自带的鸢尾花数据集
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    #计算准确率
    def accuracy_score(y_true,y_pred):
        accuracy = np.sum(y_true == y_pred,axis=0) / len(y_true)
        return accuracy
    print('-------- Classification Tree --------')
    data = datasets.load_iris()
    X = data.data
    Y = data.target
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
    for model in ["ID3","C4.5"]:
        clf = ClassDecisionTree()
        clf.fit(X_train,Y_train,model=model)
        clf.print_tree()
        y_pred = clf.predict(X_test)
        y_pred = np.array(y_pred)
        accuracy = accuracy_score(Y_test,y_pred)
        print(model + "  Accuracy:",accuracy)
        print()