# -*- coding: utf-8 -*-
# @Time    : 2019/10/4 12:50
# @Author  : Weiyang
# @File    : Example.py

#=======================================================================
# 高斯混合模型实例
# 1. 先聚类
# 2. 再分类
#=======================================================================

from GMM import GMM
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from util import *


# 构造聚类数据,X是特征数据，Y是相应的label，此时生成的是半环形图
X,Y = make_moons(n_samples=1000,noise=0.04,random_state=0)
# 划分数据，一部分用于训练聚类，一部分用于分类
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = GMM(X_train,K=10)
# 获取各个类别的概率
result = model.fit()
print('每条数据属于各个类别的概率如下: ',result)

# 获取每条数据所在的类别
label_train = np.argmax(result,axis=1)
print(label_train)

# 获取测试数据所在的类别的概率
result_test = model.predict(X_test)
# 获取测试数据的类别
label_test = np.argmax(result_test,axis=1)


# 展示原始数据分布及其label
ax1 = plt.subplot(211)
ax1.scatter(X[:,0],X[:,1],s=50,c=Y,marker='x',cmap='viridis',label="Original")
ax1.set_title('Original Data and label Distribution')

# 将聚类后的训练数据和其相应的label拼接起来展示
ax2 = plt.subplot(212)
newX = np.array(X_train.tolist() + X_test.tolist())
newY = np.array(label_train.tolist() + label_test.tolist())
plot_gmm(newX,newY,model.alpha,model.mu,model.cov,ax2)
ax2.set_title('GMM Matching Result')
plt.show()