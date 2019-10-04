# -*- coding: utf-8 -*-
# @Time    : 2019/10/4 16:16
# @Author  : Weiyang
# @File    : Example2.py

#=======================================================================
# 高斯混合模型实例
# 1. 先聚类
# 2. 再分类
#=======================================================================

from GMM import GMM
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from util import *


# 构造聚类数据,X是特征数据，Y是相应的label，此时生成的是半环形图
X, Y = make_blobs(n_samples=700, centers=4,cluster_std=0.5, random_state=2019)
# 划分数据，一部分用于训练聚类，一部分用于分类
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = GMM(X_train,K=4)
# 获取训练数据各个类别的概率
result_train = model.fit()
print('每条数据属于各个类别的概率如下: ',result_train)

# 获取训练数据所在的类别
label_train = np.argmax(result_train,axis=1)
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
ax2.scatter(newX[:,0],newX[:,1],s=50,c=newY,marker='o',cmap='viridis',label="GMM")
ax2.set_title('GMM Clustering Result')
plt.show()