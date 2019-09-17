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
    '''CART分类决策树模型'''
    def calculate_gini(self,y,y1,y2):
        '''计算Gini指数，值越小越好，Gini指数是不确定性的度量'''
        y = np.squeeze(y)
        # 计算子集y1的gini指数
        label_count1 = self._labelCounts(y1) # 子集y1中各个类别的数量
        y1_gini = 1.0
        for label in label_count1.keys():
            y1_gini -= (float(label_count1[label]) / len(y1))**2
        # 计算子集y2的gini指数
        label_count2 = self._labelCounts(y2)  # 子集y2中各个类别的数量
        y2_gini = 1.0
        for label in label_count2.keys():
            y2_gini -= (float(label_count2[label]) / len(y2)) ** 2
        # 计算集合y关于当前特征的Gini指数
        p1 = float(len(y1)) / len(y)
        p2 = float(len(y2)) / len(y)
        gini = p1 * y1_gini + p2 * y2_gini
        # 由于在DecisionTree类中调用时，采用的是值越大越好的策略，因此需要将gini添上负号
        return gini

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
        model可取"CART"
        '''
        self.impurity_calculation = self.calculate_gini

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
    clf = ClassDecisionTree()
    clf.fit(X_train,Y_train)
    clf.print_tree()
    y_pred = clf.predict(X_test)
    y_pred = np.array(y_pred)
    accuracy = accuracy_score(Y_test,y_pred)
    print("CART  Accuracy:",accuracy)


