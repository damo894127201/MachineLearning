# -*- coding: utf-8 -*-
# @Time    : 2019/9/10 10:13
# @Author  : Weiyang
# @File    : perception.py
######################################################################
# 感知机模型：f(x)=sign(w.x+b)
# 用途：二类分类的线性分类模型
# 感知机模型学习旨在求出将训练数据进行线性划分的分离超平面：wx+b=0
# 目标函数：误分类点到分类超平面的距离L(w,b)=-sum(yi(w.xi+b))
# 学习策略：使误分类点到分类超平面的距离最小
# 最优化方法：梯度下降，求出w,b,每次仅对一个误分类点进行学习
# 学习算法：
# 原始形式：f(x)=sign(w.x+b)
# 对偶形式：f(x)=sign(sum(alphaj * yj * xj . x + b))
######################################################################

import numpy as np

class Perception:
    '''感知机模型'''
    def __init__(self,r):
        self.r = r # 学习率
        self.w = None
        self.b = None

    def sign(self,x,w,b):
        '''符号函数: x 是单个点'''
        y = np.sum(np.dot(w,x)) + b
        if y > 0:
            return 1# 返回正类
        return -1 # 返回负类

    def loss(self,x,y,w,b):
        '''误分类点到分离超平面wx+b=0的距离之和：x 是 多个点的集合'''
        loss = -1 * y * self.sign(x,w,b)
        if loss > 0:
            return loss
        return 0

    def gradientDescent(self,x,y,w,b,r):
        '''梯度下降算法'''
        step = 1 # 记录迭代次数
        loss = 1 # 误分类点到分类超平面的距离，初始值设为一个正数
        while loss > 0:
            index = 0 # 用于记录误分类点的索引
            # 循环遍历所有点，如果遇到误分类点则跳出
            for i in range(len(x)):
                loss = self.loss(x[i],y[i],w,b)
                if loss > 0: # 如果当前点被误分类，则跳出循环开始更新参数w,b
                    index = i # 每次仅对一个误分类点进行学习
                    break
            if loss > 0:
                w = w - r * (-1 * np.dot(x[index],y[index])) # w 是一个多维向量
                b = b - r * (-1 * y[index]) # b 是一个标量
                print('Epoch: ',step,', wrong dot: ',x[index],', w: ',w,', b: ',b)
            step += 1
        self.w = w
        self.b = b

    def originalForm(self,x,y,w,b):
        '''感知机原始形式'''
        self.gradientDescent(x,y,w,b,self.r)

    def gram(self,x):
        '''Gram矩阵计算: 向量两两之间的内积构成的矩阵'''
        gram = np.zeros((len(x),len(x)))
        for i in range(len(x)):
            for j in range(i,len(x)):
                gram[i][j] = np.dot(x[i],x[j])
                if i != j :
                    gram[j][i] = gram[i][j]
        return gram

    def dualForm(self,x,y):
        '''感知机对偶形式'''
        step = 1
        gram_x = self.gram(x) # x的gram矩阵
        alpha = [0] * len(x) # alpha参数序列初始化为0
        b = 0 # 偏置值初始化为0
        flag = -1 # flag用来表示是否有点被误分类，如果被误分类，该值为负或0
        while flag <= 0:
            for i in range(len(x)):
                flag = y[i] * (np.sum(np.multiply(np.multiply(alpha,y),gram_x[i])) + b)
                if flag <= 0: # 如果有误分类的点
                    alpha[i] = alpha[i] + self.r
                    b = b + self.r * y[i]
                    break
            print('Epoch: ', step, ', wrong dot: ', x[i], ', alpha: ', alpha, ', b: ', b)
            step += 1
        self.w = np.dot(np.multiply(alpha,y),x)
        self.b = b
        print('w: ',w,' b:',b)

    def predict(self,x):
        '''预测函数: x 是多个点的集合，w,b是已知分离超平面的参数'''
        result = []  # 存储预测结果
        for i in x:
            y = np.sum(np.dot(self.w, i)) + self.b
            if y > 0 or y == 0:
                result.append(1)  # 返回正类
                continue
            result.append(-1)  # 返回负类
        return result

if __name__ == "__main__":
    x = [[3,3],[4,3],[1,1]]
    y = [1,1,-1]
    w = [0] * 2
    b = 0
    model = Perception(r=1)
    #model.originalForm(x,y,w,b)
    model.dualForm(x,y)
    print(model.w,model.b)
    print(model.predict([[3,3],[4,3],[1,1]]))