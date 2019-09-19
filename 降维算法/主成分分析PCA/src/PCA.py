# -*- coding: utf-8 -*-
# @Time    : 2019/9/19 13:59
# @Author  : Weiyang
# @File    : PCA.py

#========================================================================================
# 主成分分析PCA

# 是一种线性、非监督、全局的降维算法。这一方法利用正交变换把由线性相关变量表示的观测数据转换为
#      少数几个由线性无关变量表示的数据，线性无关的变量称为主成分。主成分的个数通常小于原始变量
#      的个数。空间变换只是用不同的基表示数据而已，若句子P，满足P.T*P=E(单位阵)，则P是正交矩阵
#      若句子A 经过正交矩阵P的线性变换得到新矩阵B，即B=P*A，则成B是A经过正交变换得到的。
# PCA旨在找到数据中的主成分，并利用这些主成分表征原始数据，从而达到降维的目的。
# 主成分是指数据经过正交变换后在某个方向(某个维度，不一定是原始维度)上的投影分布得更为分散，即
#         数据在这个方向上方差更大，而数据在这个维度上的投影便是主成分。降维后的数据与原始数据
#         已经不处于同一个空间。

# 主成分分析的原理：
#    首先对给定数据进行归一化，使得数据每一维度值的平均值为0。之后对数据进行
#    正交变换，原来由线性相关变量表示的数据，通过正交变换变成由若干个线性无关的新变量表示的数据。
#    新变量是可能的正交变换中变量的方差的和(信息保存)最大的，方差表示在新变量上信息的大小。将新
#    变量依次称为第一主成分、第二主成分等。通过主成分分析，可以利用主成分近似地表示原始数据，这
#    可理解为发现数据的"基本结构"，即数据中变量之间的关系；也可以把数据由少数主成分表示，这可理
#    解为对数据降维。


# PCA 的目标在于 最大化投影方差，即目标函数为投影后的方差,通过对该方差引入拉格朗日乘子，并求导取0
#     可以推知，中心化后的数据 在某个方向上投影后的方差就是其协方差与该方向的特征向量对应的特征值。
#     要求最大的方差也就是找协方差矩阵最大的特征值，最佳投影方向就是最大特征值对应的特征向量，次佳
#     投影方向位于最佳投影方向的正交空间中，是第二大特征值对应的特征向量。

# PCA 求解过程主成分以及降维后新数据的过程：
# 1. 对样本数据X进行中心化处理，即每个样本减去均值，得到中心化的样本Y,新样本的均值为0
# 2. 求新样本Y的协方差矩阵cov(Y)=E((y_i - 0)(y_j - 0))=E(y_j*y_j)
# 3. 对协方差矩阵进行特征值分解，将特征值从大到小排序
# 4. 取特征值前K大的值，对应的特征向量 w1,w2,..,wk，通过以下映射将n维的数据Y映射到K维：
#    new_samples = [w1.T * Y,w2.T * Y,...,wk.T * Y] 新的new_samples的第k维就是数据Y在第k个
#    主成分wk方向上的投影，通过选取最大的K个特征值对应的特征向量，我们将方差较小的特征(噪声)
#    抛弃，使得每个n维列向量y_i被映射为K维列向量new_samples_i
#========================================================================================

import numpy as np

class PCA:
    '''主成分分析，无监督的降维算法'''
    def __init__(self,k):
        self.k = k # 降维后数据的维度

    def _normalization(self,X):
        '''对数据中心化处理，使得均值为0'''
        if np.shape(X)[0] == 0 :
            print('----------  数据不得为空  ----------')
            exit()
        # 判断原数据的维度与降维后数据的维度大小
        old_dims = np.shape(X)[1]
        if self.k >= old_dims:
            print('-----------  请输入合适的降维后数据的维度值  --------')
            exit()
        n_samples = np.shape(X)[0] # 样本数量
        # 中心化
        X = X - np.mean(X,axis=0) # 求样本各个维度的残差，注意是同一维度
        return X

    def _calculate_covariance_matrix(self,X):
        '''求协方差矩阵'''
        X = self._normalization(X) # 中心化数据
        n_samples = np.shape(X)[0] # 样本数据
        cov = 1 / n_samples * np.matmul(X.T,X) # 协方差矩阵
        return cov,X

    def fit(self,X):
        '''降维,X = np.array([[value,..],..])'''
        # 获取协方差矩阵和中心化后的X
        cov_matrix,new_X = self._calculate_covariance_matrix(X)
        # 矩阵特征值分解：特征值 ，特征向量
        features,feature_vectors = np.linalg.eig(cov_matrix)

        # 对特征值从大到小排序，并取前self.k个最大的特征值
        # 该函数会对features数组排序，返回的是从小到大排序后原数组中各个值在新序列中的索引位置
        # [::-1] 将位置逆序，变为从大到小
        idx = features.argsort()[::-1]
        # 按特征值从大到小排序后，各个特征值对应的特征向量的新顺序
        # 注意: 假设原矩阵为A ，特征向量w和特征在 lambd，满足 A*w = labmd * w
        # A*[w1,w2,w3] = [lambd1*w1,lambd2*w2,lambd3*w3,..,lambdk * wk]
        # A*w 即为原矩阵在某个特征向量上的投影，因此在返回的特征向量矩阵中，每个列代表着一个特征向量
        feature_vectors = feature_vectors[:,idx]
        # 获取前self.k个特征向量
        k_vectors = feature_vectors[:,:self.k]

        # 降维数据 即 A * [w1,w2,..,wk]
        return np.matmul(new_X,k_vectors)

if __name__ == "__main__":
    from sklearn import datasets
    import matplotlib.pyplot as plt

    # 加载数据
    data = datasets.load_iris()
    X = data.data # 样本数据
    y = data.target # 样本的类别

    # 降维到2维
    X_trans = PCA(k=2).fit(X) # 降维到2维
    x1 = X_trans[:, 0] # 获取第一维度的值
    x2 = X_trans[:, 1] # 获取第二维度的值

    cmap = plt.get_cmap('viridis') # 序列色彩图
    # 不同的类别用不同的颜色
    colors = [cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]

    class_distr = []
    # Plot the different class distributions
    labels = np.unique(y)
    for i, l in enumerate(labels):
        _x1 = x1[y == l] # 获取当前类别的第一个维度值
        _x2 = x2[y == l] # 获取当前类别的第二个维度值
        class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))

    # Add a legend
    plt.legend(class_distr, labels, loc=1)

    # Axis labels
    plt.suptitle("PCA Dimensionality Reduction")
    plt.title("Iris Dataset")
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    # 降维到3维
    from mpl_toolkits.mplot3d import Axes3D

    X_trans = PCA(k=3).fit(X)  # 降维到3维
    x1 = X_trans[:, 0]  # 获取第一维度的值
    x2 = X_trans[:, 1]  # 获取第二维度的值
    x3 = X_trans[:, 2] # 第三个维度值

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('viridis')  # 序列色彩图
    # 不同的类别用不同的颜色
    colors = [cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]

    class_distr = []
    # Plot the different class distributions
    labels = np.unique(y)
    for i, l in enumerate(labels):
        _x1 = x1[y == l]  # 获取当前类别的第一个维度值
        _x2 = x2[y == l]  # 获取当前类别的第二个维度值
        _x3 = x3[y == l]
        class_distr.append(ax.scatter(_x1, _x2, _x3,color=colors[i]))

    # Add a legend
    plt.legend(class_distr, labels, loc=1)

    # Axis labels
    plt.suptitle("PCA Dimensionality Reduction")
    plt.title("Iris Dataset")
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.show()