# -*- coding: utf-8 -*-
# @Time    : 2019/11/7 8:20
# @Author  : Weiyang
# @File    : PCA_SVD.py

#======================================================================================================================
# 主成分分析PCA：基于原始矩阵的奇异值分解算法

# 基于协方差矩阵的特征值分解算法
# 定理：这是个有约束的优化问题，可通过构造拉格朗日函数来证明
# 设X是m维随机变量，∑是X的协方差矩阵(相关矩阵)，∑的特征值分别是λ1>=λ2>=λ3>=...>=λm>=0，特征值对应的单位特征向量分别是
# alpha1,alpha2,...,alpha_m，则X的第 k 主成分是：
#                                     y_k = (alpha_k)^T * X = alpha_1k * x1 + alpha_2k * x2 +...+ alpha_mk * x_m
# X的第 k 主成分的方差是：
#                                     var(y_k) = (alpha_k)^T * ∑ * alpha_k = (alpha_k)^T * λk * alpha_k = λk
# 即协方差矩阵∑的第k个特征值。

# 基于矩阵的奇异值分解算法实现PCA
# 1. 输入：m * n 样本矩阵X，其每一行元素的均值为0，方差为1(因此需要对原始矩阵的每一个维度进行规范化),
#          也可以只进行中心化处理，即每个维度的变量只减去其均值，但不除以方差，具体看效果
# 2. 输出：k * n 样本主成分矩阵Y ，使用时再转置一下，就是n个样本的k个主成分的矩阵了
# 3. 参数：主成分的个数 k
# 4. 构造新的 n*m 矩阵 X' = 1/sqrt(n-1) * X^T ,X'的每一列均值为0
# 5. 对矩阵X'进行截断奇异值分解，得到
#                        X' = U * ∑ * V^T
#    有k个奇异值，及其对应的奇异向量，矩阵V的前k列构成k个主成分
# 6. 求 k*n 样本主成分矩阵
#                        Y = V^T * X
# 7. 输出时，需对Y转置一下，变成每行一条数据的样式

# 证明过程如下：
# 1. X' = 1/sqrt(n-1) * X^T      =>   (X')^T * X' = (1/sqrt(n-1) * X^T)^T * 1/sqrt(n-1) * X^T = 1/(n-1) * X * X^T
# 2. (X')^T * X' 正好等于 X 的 样本协方差矩阵 S_x
# 3. 而求X'奇异值分解的过程的第一步，便是求 (X')^T * X' 的特征值和特征向量，而这些特征值和特征向量 正好是矩阵X样本协方差矩阵的
#    特征值和特征向量，因此 矩阵X 主成分分析归结于对 矩阵X' 做 奇异值分解，即 X' = U * ∑ * V^T 的V矩阵，其列向量正好是
#    矩阵X样本协方差矩阵 的特征向量， X'的奇异值正好是 矩阵X样本协方差矩阵 的特征值
#======================================================================================================================

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
        # 规范化，即均值为0，方差为1
        #X = (X - np.mean(X, axis=0))/np.var(X,axis=0)
        return X

    def fit(self,X):
        '''降维,X = np.array([[value,..],..])'''
        # 对矩阵X进行中心化处理，即每列均值为0
        X = self._normalization(X)
        # 转置一下，变为每列为一条数据
        X = X.T

        # 构建新矩阵
        # X' = 1/sqrt(n-1) * X^T
        n_samples = X.shape[0] # 样本个数
        new_X = 1/np.sqrt(n_samples-1) * X.T

        # (X')^T * X'矩阵特征值分解：特征值 ，特征向量
        features,feature_vectors = np.linalg.eig((new_X.T).dot(new_X)) # feature_vectors中每一列为一个特征向量

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

        # 降维数据 即 [w1,w2,..,wk]^T * A
        return np.matmul(k_vectors.T,X).T

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
    plt.suptitle("PCA_SVD Dimensionality Reduction")
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
    plt.suptitle("PCA_SVD Dimensionality Reduction")
    plt.title("Iris Dataset")
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.show()