# -*- coding: utf-8 -*-
# @Time    : 2019/10/3 19:40
# @Author  : Weiyang
# @File    : GMM.py

#==================================================================================================================
# 高斯混合模型(Gaussian Mixture Model,GMM)：聚类算法，生成模型，采用EM算法实现
# 含义：高斯混合模型是用高斯概率密度函数(正态分布密度函数)精确地量化事物，将一个事物分解为若干个基于
#       高斯概率密度函数线性组合的模型。事物的数学表现形式为曲线，高斯混合模型的意思就是，任何一个曲线，无论多么复杂，
#       都可以用若干个高斯曲线来无限逼近它，这便是高斯混合模型的基本思想。

# 数学形式：
#        高斯混合模型是指具有如下形式的概率分布模型：
#                                P(y|θ) = ∑{k=1,...,K}α_k * φ(y|θk)
#        其中，α_k是系数，α_k >= 0，∑{k=1,...,K}α_k = 1；φ(y|θk)是高斯分布密度，θk = (μ_k,(σ_k)^2)
#                               φ(y|θk) = 1 / (sqrt(2π)*σ_k) * exp(- (y - μ_k)^2 / 2(σ_k)^2)
#        称为第 k 个分模型。
#        如果得到高斯混合模型各个参数的值，我们就可以预测各个数据所属的类别的概率。

# 一般混合模型可以由任意概率分布密度替代高斯分布密度，这里只介绍最常用的高斯混合模型。

# 用于聚类：高斯混合模型假设每个类簇的数据都是符合高斯分布的，当前数据呈现的分布就是各个类簇的高斯分布叠加在一起的结果。

# 用于生成：可以通过估计各个分模型的概率密度函数来生成新数据；

# GMM模型因其优秀的聚类表现，以及可以生产样本的强大功能，在风控领域的应用非常广泛。
# 如对反欺诈中的欺诈样本抓取与生成、模型迭代中的幸存者偏差等问题都有一定的作用。
# 比如说在反欺诈检测模型中，可以先通过GMM模型对欺诈样本进行聚类，后将聚类后得到的簇作为同一类欺诈手段，
# 后续只针对不同的簇进行建模，在实践中对欺诈客户的召回有很好的效果。

# 与Kmeans聚类算法比较
# 相同点：
#       1. 都是聚类算法；
#       2. 都需要指定聚类的簇K值
#       3. 都往往只能收敛于局部最优值
#          避免局部最优的做法是 选取几个不同的初值进行迭代，然后对得到的各个估计值加以比较，从中选择最好的；
# 不同点：
#       1. Kmeans算法无法给出一个样本属于某类的概率，高斯混合模型可以
#       2. 二维的Kmeans模型的本质是，它以每个簇的中心为圆心，簇中的点到簇中心点的欧式距离最大值为半径画一个圆。这个圆硬性
#          地将训练集进行截断，而且，Kmeans要求这些簇的形状必须是圆形的，因此kmeans模型拟合出来的簇(圆形)与实际数据分布
#          (可能是椭圆)差别很大，经常出现多个圆形的簇混在一起，相互重叠。
#       3. 高斯混合模型可以拟合任意形状的数据分布，而kmeans只能拟合圆形或超球体型的数据分布；
#       4. Kmeans算法会截断数据，它是通过硬截断数据来分割类别的；GMM则是通过高斯分布来估计数据的分布，其本质上不是聚类算法，
#          而是得到一个能生成当前样本形式的分布，更像回归拟合；

# 期望极大算法(Expectation Maximization Algorithm,EM)：迭代算法，比如梯度下降算法
# 用途：用于含有隐变量(hidden variable)的概率模型参数的极大似然估计，或极大后验估计
# 引入：概率模型有时既含有观测变量(observable variable)，又含有隐变量或潜在变量(latent variable)。如果概率模型的变量都是
#       观察变量，那么给定数据，可以直接用极大似然估计法，或贝叶斯估计法(在极大似然估计法的分子分母上都加上一个常量，
#       防止估计的概率为0)；但是，当模型含有隐变量时，就不能简单地使用这些估计方法，需要使用EM算法。
# 目标函数：极大化观测数据(不完全数据)Y关于参数θ的对数似然函数
#           这个问题没有解析解，只能通过迭代求解(设定目标函数，参数更新表达式，迭代终止条件，然后赋予参数初值，开始迭代计算)

# 有两个对数似然函数：
# 1. 观测数据Y关于参数θ的对数似然函数
# 2. 完全数据(Y,Z)关于参数θ的对数似然函数，其中Z是隐变量，也是未观测数据

# EM算法每次迭代由两步组成：
# 1. E步：求期望
#        求期望，就是确定Q(θ,θ^(i))函数；
#        Q函数：完全数据的对数似然函数logP(Y,Z|θ)关于在给定观测数据Y和当前参数θ^(i)下对未观测数据Z的条件概率分布P(Z|Y,θ^(i))
#               的期望，即 Q(θ,θ^(i)) = ∑[P(Z|Y,θ^(i))*logP(Y,Z|θ)] = Ez[logP(Y,Z|θ)|Y,θ^(i)]
# 2. M步：求极大
#        求极大，就是确定Q(θ,θ^(i)) 对θ的极大值，用于更新θ

# EM算法注意点：
# 1. EM算法对初值是敏感的
# 2. 每次迭代使似然函数值增大或达到局部极值
# 3. 给出迭代停止条件，若满足 相邻两次迭代之间参数的差值或 Q函数的差值小于指定阈值，则停止迭代
# 4. EM算法通过不断求解下界的极大化逼近求解对数似然函数极大化的算法，与最大熵模型的改进的迭代尺度算法思想一致，具体而言
#    就是 通过求解使得下界(这个下界是比较紧的)取得极值的参数，而这个参数同时也是对数极大似然函数的参数，来达到间接提升
#    对数极大似然函数值(最大熵模型目标函数是 条件熵)的目的。这样，每次迭代都逐步提升下界的极值，间接也使得目标函数的值达到提升。
#==================================================================================================================

import numpy as np
from scipy.stats import multivariate_normal # 由于输入是多元的，因此用多元正态随机变量

class GMM:
    '''高斯混合模型：聚类算法，EM实现'''
    def __init__(self,X,K,max_iters=1000,threshold=1e-3):
        self.X = X # 特征向量列表,np.array([[value,value,...],...])
        self.K = K # 聚类的类簇数
        self.max_iters = max_iters # 最大迭代次数
        self.threshold = threshold # 迭代阈值，用于判定是否收敛
        self.alpha = None # 存储各个分模型的系数
        self.mu = None # 存储各个分模型中各个特征分量的均值
        self.cov = None # 粗糙各个分模型的协方差矩阵

    def _rowNormalize(self,X,axis=-1,order=2):
        '''
        对数据进行归一化，将各维度的值缩放在[0,1]之间，X = np.array([[value1,value2,...],..])
        对行进行归一化，即以整行作为计算单元
        '''

        # 以X的最后一维度为准，求X中每条数据构成的向量的L2范数
        l2 = np.atleast_1d(np.linalg.norm(X,order,axis))
        l2[l2 == 0] = 1 # 对L2范数为0的，赋值为1，避免分母为0
        return X / np.expand_dims(l2,axis)

    def _colNormalize(self,X):
        '''
        min-max标准化，使数据落在[0,1]之间
        对列进行归一化，即以整列为计算单元
        '''
        Y = np.zeros(X.shape)
        for i in range(Y.shape[1]):
            max_ = X[:,i].max() # 当前列最大的值
            min_ = X[:,i].min() # 当前列最小的值
            Y[:,i] = (X[:,i] - min_) / (max_ - min_) # 归一化
        return Y

    def _gaussianProbabilityDensity(self,X,mean,cov):
        '''
        多元高斯随机变量的概率密度函数
        mean: 均值 ；cov：协方差矩阵
        '''
        norm = multivariate_normal(mean=mean,cov=cov)
        return norm.pdf(X) # pdf是概率密度函数,得到X中每条数据对应的概率

    def _isConvergence(self,W1,W2):
        '''用于判断是否所有的wi都收敛，W1是更新前的值，W2是更新后的值'''
        for i in range(len(W1)):
            for j in range(len(W1[0])):
                if np.abs(W1[i][j] - W2[i][j]) > self.threshold:
                    return False
        return True

    def EM(self,normalization=False):
        '''EM算法实现GMM'''
        if normalization == "row":
            # 对行做归一化
            self.X = self._rowNormalize(self.X)
        elif normalization == "col":
            # 对列做归一化
            self.X = self._colNormalize(self.X)
        elif normalization == False: #不做归一化
            pass

        # 参数初始化
        n_samples,n_features = self.X.shape
        #shape = (self.K,)
        alpha = np.ones(self.K) / self.K #  初始化每个分模型的权重，初始值相同
        # shape = (self.K,n_features)
        mu = np.random.rand(self.K,n_features) # 初始化每个分模型的均值向量，由于是多元正态分布，因此每个特征对应一个均值
        last_mu = mu[:] # 存储上一轮的均值，用于判断是否收敛
        # shape = (self.K,n_features,n_features)
        cov = np.array([np.eye(n_features)] * self.K) # 初始化每个分模型的协方差矩阵，由于是多元正态分布，
                                                      # 且我们假设各个特征分量之间不相关，因此协方差矩阵是一个对角阵，对角线上
                                                      # 每个元素为对应特征分量的方差
        # shape = (n_samples,self.K)
        gama = np.zeros((n_samples,self.K)) # 存储每个分模型对每条数据的响应度
                                            # 即，当前数据来自于某个分模型的概率，要聚类self.K个簇，因此有self.K个分模型
                                            # 有n_samples个数据，因此有n_samples行；我们最终以此来判定每条数据具体属于哪个类
        # 开始迭代
        for i in range(1,self.max_iters+1):
            # E步，求期望：经过公式推导后，简化为:依据当前模型参数，计算分模型k对观测数据y_j的响应度
            # shape = (n_samples,self.K)
            prob_k = np.zeros((n_samples,self.K)) # 存储样本来自每个分模型的概率
            # 遍历每个分模型
            for k in range(self.K):
                # 计算每个样本来自当前分模型的概率
                prob_k[:,k] = alpha[k] * self._gaussianProbabilityDensity(self.X,mu[k],cov[k])
            # 分别计算每个样本来自所有模型的概率之和，来求响应度
            sum_prob = np.sum(prob_k,axis=1) # shape = (n_samples,)
            # 计算响应度,shape = (n_samples,self.K)
            gama = prob_k / np.expand_dims(sum_prob,axis=1) # 保存维度一致

            # M步，求极大：经过公式推导后，计算新一轮迭代的参数，分模型的均值mu 和 协方差cov，每个分模型的系数alpha

            # 注意,对于多元正态分布，均值mu 和 方差 σ都是多元的,由于我们假设各个特征分量之间不相关，即相关系数为0
            # 因此可以根据各个特征分量的方差求得协方差矩阵

            # shape = (self.K,)
            sum_gama = np.sum(gama,axis=0) # 所有样本来自于同一个分模型的响应度之和

            # 更新alpha，每个分模型的系数，shape = (self.K,)
            alpha = sum_gama / n_samples

            # 更新每个分模型的均值和协方差
            for k in range(self.K):
                # self.X.shape = (n_samples,n_features),gama.shape = (n_samples,self.K)
                # 更新当前分模型中，每个特征分量的均值，因此使用gama[:,[k]]，使得每一个元素是一个单值数组，便于广播计算
                gama_X = np.multiply(self.X,gama[:,[k]]) # ∑gama_jk * Xj ,shape=(n_samples,n_features)
                mu[k] = np.sum(gama_X,axis=0) / sum_gama[k] # shape = (self.K,n_features)

                # 更新协方差，首先得更新当前分模型中，每个特征分量的方差
                # 将self.X的每一行数据，其各个维度减去相应维度的均值
                # X_mu.shape = (n_samples,n_features)
                X_mu = self.X - mu[k] # mu[k].shape = (n_features,) self.X.shape=(n_samples,n_features)
                gama_X_mu = np.multiply(gama[:,[k]],X_mu) # shape=(n_samples,n_features)
                '''
                # shape = (n_features,n_features)
                numerator = np.dot(np.transpose(gama_X_mu),X_mu) # (gama_i * (X_i-mu_k)).T * (X_i-mu_k)
                # 更新当前分模型中，协方差
                #cov[k] = numerator / sum_gama[k] # shape=(n_features,n_features)
                '''
                # shape = (n_samples,n_features)
                numerator = np.multiply(gama_X_mu,X_mu)
                # 更新当前分模型每个特征分量的方差
                variance = np.sum(numerator,axis=0) / sum_gama[k] # shape=(n_features,)
                # 要避免cov协方差矩阵不可逆，由于每个特征分量的方差是作为协方差矩阵的对角元素，因此不能为0
                variance[np.where(variance == 0)] = 0.00001 # 将方差为0的设置为一个极小值，避免协方差矩阵不可逆
                # 由于协方差矩阵是将各个特征分量的方差放置于矩阵对角线上，因此更新当前分模型的协方差
                cov[k] = np.diag(variance)

            # 判断是否收敛
            if self._isConvergence(last_mu,mu) and i > 0.5 * self.max_iters:
                return gama,alpha,mu,cov

        return gama, alpha, mu, cov # 每个样本对每个分模型的响应度矩阵，每个分模型的系数，
                                    # 每个分模型的每个特征分量的均值，分模型的协方差矩阵

    def fit(self,normalization=False):
        '''对数据进行聚类，并返回聚类结果'''
        gama,self.alpha,self.mu,self.cov = self.EM(normalization)

        return gama.round(2) # 每个样本对每个分模型的响应度，即属于某个类的概率

    def predict(self,X):
        '''对新数据进行预测，预测其所属的类别，X = np.array([[value,...],...])'''
        # shape = (n_samples,self.K)
        prob_k = np.zeros((len(X), self.K))  # 存储样本来自每个分模型的概率
        # 遍历每个分模型
        for k in range(self.K):
            # 计算每个样本来自当前分模型的概率
            prob_k[:, k] = self.alpha[k] * self._gaussianProbabilityDensity(X, self.mu[k], self.cov[k])
        # 分别计算每个样本来自所有模型的概率之和，来求响应度
        sum_prob = np.sum(prob_k, axis=1)  # shape = (n_samples,)
        # 计算响应度,shape = (n_samples,self.K)
        gama = prob_k / np.expand_dims(sum_prob, axis=1)  # 保存维度一致

        return gama.round(2)

    def sample(self,n_samples=200,random_state=0):
        '''依据当前高斯混合模型，生成新数据'''
        # 创建一个空的多维数组，用于容纳生成的数据，事实上这个数组内的数据是随机生成的
        data = np.empty((n_samples,self.mu.shape[1]))
        # TO DO
        pass