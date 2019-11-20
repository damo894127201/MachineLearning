# -*- coding: utf-8 -*-
# @Time    : 2019/11/17 17:54
# @Author  : Weiyang
# @File    : LDA.py

#======================================================================================================================
# 线性判别分析(Linear Discriminant Analysis,LDA): 有监督的降维算法和分类算法
# LDA由Fisher在1936年发明，因此也称为Fisher LDA

# 原理：最大化类间距离(用类间散度矩阵S_B度量)和最小化类内距离(用类内散度矩阵S_w度量)，目的在于使投影后两类之间的距离尽可能大

# 与PCA的区别：PCA是无监督的降维算法，因此PCA在降维时没有考虑数据的类别，只是把原数据映射到一些方差比较大的方向上而已，因此
#              PCA有时降维后的数据再进行分类后，效果会非常差；LDA在降维时充分考虑了数据的类别，它本身就是为分类服务的，在有
#              多个类别的数据时，其目标是找到一个投影方向ω，使得投影后的样本尽可能按照原始类别分开；

# 与PCA的相同处：在降维过程中，都使用了矩阵的特征分解：PCA是对规范化的样本协方差矩阵做特征分解，LDA则是对 S_w的逆阵*S_B做特征分解；
#                相同之处在于：特征分解后的特征值以及其特征向量，特征值越大的，对应的特征向量，对于PCA是其最主要的主成分的投影方向ω
#                用ω*原始数据矩阵A，便得到第一个主成分，这里ω是一个行向量，也可以用列向量表示，不过要转置一下；对于LDA来说，
#                最佳的投影方向ω便是最大特征值对应的特征向量；当想将数据降维到k维时，只需要选择前k个最大的特征向量，分别左乘
#                原始矩阵即可得到降维后的各个维度的数据；

# 二类LDA算法的推导：
# 基于LDA的原理，假设有两个类别，投影方向为ω，其中ω是单位向量，D1 和 D2 分别表示按照ω投影后的两个类别数据的方差，则有
# 1. 类间距离(投影之后两个类别之间的距离)：
#                                D(C1,C2) =  ||ω^T(μ1 - μ2)||^2 (F2范数，各个元素的平方之和，不是矩阵平方)
#                                其中，μ1是类别1投影之前的均值，μ2是类别2投影之前的均值
#                                ω^T*μ1 是类别1投影之后的均值，ω^T*μ2 是类别2投影之后的均值
#                                ω^T*x 是数据x(向量)投影之后值，x是列向量，μ也是列向量
# 2. 类内距离(整个数据集的类内方差)：
#    类内方差定义为各个类别数据的方差之和
#                                D = D1 + D2
#                                D1 = ∑_{x∈类别1}(ω^T*x - ω^T*μ1)^2 = ∑_{x∈类别1}ω^T*(x - μ1)*(x - μ1)^T * ω
#                                D2 = ∑_{x∈类别2}(ω^T*x - ω^T*μ2)^2 = ∑_{x∈类别2}ω^T*(x - μ2)*(x - μ2)^T * ω
#                                x 是原始数据
# 3. 定义目标函数：
#    类间距离和类内距离的比值，目标在于求这个比值的最大值
#                                max J(ω) = ||ω^T(μ1 - μ2)||^2 / (D1 + D2)
#                                          = ω^T*(μ1 - μ2)*(μ1 - μ2)^T * ω / (∑_{x∈Ci}ω^T*(x - μi)*(x - μi)^T * ω)
#    1. 定义类间散度矩阵S_B：
#                                S_B = (μ1 - μ2)*(μ1 - μ2)^T
#    2. 定义类内散度矩阵S_w：
#                                S_w = ∑_{x∈Ci}(x - μi)*(x - μi)^T
#       注意 类间散度矩阵与类间距离的差异，类内散度矩阵与类内距离的差异
#    3. 目标函数J(ω)可改写为：
#                                J(ω) = ω^T * S_B * ω / (ω^T * S_w * ω)
#       值得注意的是分子ω^T * S_B * ω 与 分母 ω^T * S_w * ω 都是标量，是一个数
#    4. 求使目标函数J(ω)最大的投影方向ω，方法：对ω求偏导，再使偏导为0即可
#                                ∂J(ω)/∂ω = 0
#                          可得
#                                (ω^T * S_w * ω)S_B * ω = (ω^T * S_B * ω)S_w * ω
#                          又由于 J(ω) = ω^T * S_B * ω / (ω^T * S_w * ω)
#                          我们可令J(ω) = λ,则上式变为
#                          S_B * ω = λ * S_w * ω ，继续化简
#                          (S_w)^-1 * S_B * ω = λω
#       从这里我们可以看出，我们最大化的目标是对应了一个矩阵的特征值，投影方向变为了特征值对应的特征向量，于是LDA降维变成了
#       一个求矩阵特征向量的问题，J(ω)就对应了矩阵 (S_w)^-1 * S_B 最大的特征值，而投影方向就是这个特征值对应的特征向量。
#    5. LDA最大能降维的维数
#                          当总共有K个类别时，LDA降维后数据的维度最大值为K-1维，而PCA没有这个限制
# 4. LDA算法流程
#    1. 输入：数据集D={(x1,y1),...,(xm,ym)}，其中任意样本x_i为n维向量，y_i∈{C1,C2,...,Ck}，降维到d维
#    2. 输出：降维后的样本集D'
#    3. 计算类内散度矩阵S_w
#    4. 计算类间散度矩阵S_B
#    5. 计算矩阵(S_w)^-1 * S_B
#    6. 计算矩阵(S_w)^-1 * S_B的最大的d个特征值和对应的d个特征向量(w1,w2,...,wd),得到投影矩阵W，其中每列是一个特征向量
#    7. 对样本集中的每个样本的特征向量x_i，计算降维后的特征向量 z_i = W^T * x_i
#    8. 得到输出的降维后的样本集D' ={(z1,y1),...,(zm,ym)}

# LDA 用于分类
#    LDA分类基本思想是假设各个类别的样本数据符合高斯分布，这样利用LDA进行投影后，
#    可以利用极大似然估计计算各个类别投影数据的均值和方差，进而得到该类别高斯分布的概率密度函数。
#    当一个新的样本到来后，我们可以将它投影，然后将投影后的样本特征分别带入各个类别的高斯分布概率密度函数，
#    计算它属于这个类别的概率，最大的概率对应的类别即为预测类别。

# 多类LDA原理
# 假设数据集D={(x1,y1),...,(xm,ym)}，其中任意样本xi为n维向量，yi∈{C1,...,Ck}。我们定义Nj(j=1,2,...,k)为第j类样本的个数，
# Xj(j=1,2,...,k)第j类样本的集合，而μj(j=1,2,...,k)为第j类样本的均值向量，定义∑j(j=1,2,...,k)为第j类样本的协方差矩阵。
# 在二类LDA里面定义的公式很容易推到多类LDA
# 由于我们是多类向低维投影，则此时投影到的低维空间就不是一条直线，而是一个超平面。假设我们投影到的低维空间的维度为d，对应的基向量
# 为(ω1,ω2,...,ωd),基向量组成的投影矩阵为W，它是一个n*d的矩阵
# 此时我们的优化目标应该可以为：
#                            J(W) = W^T * S_B * W / (W^T * S_w * W)
#                  其中
#                           S_B = ∑_{j=1,..,k}Nj*(μj - μ)*(μj - μ)^T
#                           S_w = ∑_{j=1,..,k}S_wj = ∑_{j=1,..,k}∑_{x∈Xj}(x - μj)*(x - μj)^T
#                           μ为所有样本均值向量
# 但有一个问题，就是 W^T * S_B * W 和 (W^T * S_w * W)都是矩阵，不是标量，无法作为一个标量函数来优化！也就是说，我们无法直接用
# 二类LDA的优化方法，怎么办呢？一般来说，我们可以用其它的一些替代优化目标来实现，常见的一个多类LDA优化目标函数定义为：
#                           argmax_{W} J(W) = ∏_diag  W^T * S_B * W / (∏_diag W^T * S_w * W)
#                  其中
#                           ∏_diag A 为A的主对角线元素的乘积，W为n*d的矩阵
# J(W)的优化过程可以转化为：
#                           J(W) = ∏_{i=1,..,d} ωi^T * S_B * ωi / (∏_{i=1,..,d} ωi^T * S_w * ωi)
#                                = ∏_{i=1,..,d} (ωi^T * S_B * ωi / (ωi^T * S_w * ωi))
# 仔细观察上式右边，这不就是广义瑞利商嘛，最大值是矩阵(S_w)^-1 * S_B的最大特征值，最大的d个值的乘积就是矩阵(S_w)^-1 * S_B
# 的最大的d个特征值的乘积，此时对应的矩阵W为这最大的d个特征值对应的特征向量张成的矩阵。因此我们只需如同二类LDA那样对(S_w)^-1 * S_B
# 做特征分解即可，不多注意多类LDA 的S_B 和 S_w 的计算方式还是与二类LDA有所区别的。

# 广义瑞利商参见../post内博客介绍，其中，对于(B^-1/2)*A*(B^-1/2)可以做变化，即
#                          (B^-1/2)*A*(B^-1/2) = (B^-1/2)*A*(B^-1/2)*(B^1/2) / (B^1/2)
#                                              = (B^-1/2)*A / (B^1/2)
#                                              = (B^-1/2)*(B^-1/2)*A / ((B^-1/2)*(B^1/2))
#                                              = (B^-1)*A

# 多类LDA投影维度的限制
# 由于W是一个利用了样本的类别得到的投影矩阵，因此它降维到的维度d最大值为k-1，比如二类的数据，最多降维到1个维度;五类数据，最多
# 降维到4个维度，意思是降维后的数据维度最大值为4，可以是1,2,3个维度
# 为什么最大维度不是类别数k呢？因为S_B中每个μj - μ的秩为1，因此协方差矩阵相加后最大的秩为k(矩阵的秩小于等于各个相加矩阵的
# 秩的和)，但是由于如果我们知道前k-1个μj后，最后一个μk可以由前k-1个μj线性表示，因此S_B的秩最大为k-1，即特征向量最多有k-1个。


# 二类LDA与多类LDA的比较：
# 1. 相同处：计算类内散度矩阵的方式是相同的，都是各个类别之间的数据与其类均值向量：(x - μj)*(x - μj)^T的矩阵，然后再加和
#            注意上述不是距离，不是距离；均值向量是投影之前的均值向量
# 2. 不同处：计算类间散度矩阵的方式不同：
#            1. 二类LDA计算类间散度矩阵，是用投影之前的两个类的均值向量，求其(μ1 - μ2)*(μ1 - μ2)^T
#            2. 多类LDA计算类间散度矩阵，是用投影之前的每个类的均值向量，求其与投影之前全体数据的均值向量μ的：
#               Nj * (μj - μ)*(μj - μ)^T ，然后再加和得到
#======================================================================================================================

import numpy as np
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')

class LDA(object):
    '''多类LDA线性判别分析：用于降维'''

    def __init__(self,k):
        self.k = k # 降维后的维度

    '计算每个类别的类内散度矩阵'
    def compute_Sw(self,data):
        '''
        data: np.ndarray ,每列为一条数据
        data.shape = (n,m), m是当前类别的数据个数，n是每条数据的维度，即每条数据的特征个数'''
        n,m = data.shape # n个维度，m条数据
        mean = np.mean(data,axis=1) # 均值向量，按行求
        mean = mean[:,None] # 最后一维度增加一维，转为列向量,shape=(n,1)
        sw = 0
        # 遍历每条数据
        for i in range(m):
            sw += (data[:,i] - mean).dot((data[:,i] - mean).T)
        return sw # shape=(n,n)

    '计算两个类的类间散度矩阵'
    def compute_Sb(self,data1,data2):
        '''
        data1 和 data2 : np.ndarray ,每列为一条数据
        data1.shape = (n,m), m是当前类别的数据个数，n是每条数据的维度，即每条数据的特征个数；data2类同
        '''
        mean1 = np.mean(data1,axis=1)
        mean1 = mean1[:,None] # shape=(n,1)
        mean2 = np.mean(data2, axis=1)
        mean2 = mean2[:, None]
        sb = (mean1 - mean2).dot((mean1 - mean2).T) # shape=(n,n)
        return sb

    '用LDA对数据降维'
    def fit(self,data):
        'data: np.ndarray data: 待降维的数据，其中每行是一条数据，最后一维度是类别'
        logger = logging.getLogger('Train')
        m,n = data.shape # m条数据,每条数据由n个维度，其中最后一维度是类别
        # 将数据按类别划分
        dataDict = defaultdict(list)
        # 遍历数据
        for i in range(m):
            dataDict[data[i,-1]].append(data[i,:-1]) # 去除类别标签
        num_label = len(dataDict) # 数据中的类别数

        # 判断是二类LDA，还是多类LDA
        if num_label == 2:
            # 判断降维后的维度是否超过指定维度
            if self.k > n:
                logger.info('无法降维到{}维，降维后的数据的维度最大值为{}维'.format(self.k,n))
                return None,None
            # 计算类内散度矩阵
            Sw = 0
            temp = []
            for label in dataDict.keys():
                class_data = np.array(dataDict[label]) # 将相应类别的数据转为np.ndarray
                class_data = class_data.T # 转为每列为一条数据
                Sw += self.compute_Sw(class_data)
                temp.append(class_data)
            # 计算类间散度矩阵
            SB = self.compute_Sb(temp[0],temp[1])

            # 对(Sw)^-1 * S_B做特征分解，feature_vectors列向量为特征向量
            feature_value,feature_vectors = np.linalg.eig(np.mat(Sw).I.dot(SB))
            # np.linalg.eigh 适用于对称阵的特征值分解，返回的特征值严格升序排列的
            # 对特征值从大到小排序，返回索引,由于np.argsort是按照升序排序的，因此对排序的对象添加一个负号来达到降序的目的
            index_vec = np.argsort(- feature_value)
            feature_vectors = feature_vectors[:,index_vec] # 降序排列特征向量
            W = feature_vectors[:,:self.k] # 投影矩阵
            # 对数据降维
            labels = data[:,-1] # 存放label
            new_data = data[:,:-1].T # 将data转为每列一条数据
            new_data = W.T.dot(new_data)
            new_data = new_data.T # 将data转为每行一条数据
            new_data = np.concatenate((new_data,labels[:,None]),axis=1) # 加上label
            return new_data,W
        elif num_label > 2:
            # 判断降维后的维度是否超过指定维度
            if self.k > (num_label - 1):
                logger.info('无法降维到{}维，降维后的数据的维度最大值为{}维'.format(self.k,num_label-1))
                return None,None
            # 计算类内散度矩阵
            Sw = 0
            temp = []
            for label in dataDict.keys():
                class_data = np.array(dataDict[label]) # 将相应类别的数据转为np.ndarray
                class_data = class_data.T # 转为每列为一条数据
                Sw += self.compute_Sw(class_data)
                temp.append(class_data)
            # 计算类间散度矩阵
            SB = 0
            for data_i in temp:
                SB += len(data_i) * self.compute_Sb(data_i,data[:,:-1].T)

            # 对(Sw)^-1 * S_B做特征分解，feature_vectors列向量为特征向量
            feature_value,feature_vectors = np.linalg.eig(np.mat(Sw).I.dot(SB))
            # np.linalg.eigh 适用于对称阵的特征值分解，返回的特征值严格升序排列的
            # 对特征值从大到小排序，返回索引,由于np.argsort是按照升序排序的，因此对排序的对象添加一个负号来达到降序的目的
            index_vec = np.argsort(- feature_value)
            feature_vectors = feature_vectors[:,index_vec] # 降序排列特征向量
            W = feature_vectors[:,:self.k] # 投影矩阵
            # 对数据降维
            labels = data[:,-1] # 存放label
            new_data = data[:,:-1].T # 将data转为每列一条数据
            new_data = W.T.dot(new_data)
            new_data = new_data.T # 将data转为每行一条数据
            new_data = np.concatenate((new_data,labels[:,None]),axis=1) # 加上label
            return new_data , W

if __name__ == '__main__':
    from sklearn.datasets.samples_generator import make_classification
    import matplotlib.pyplot as plt
    from pylab import mpl

    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题
    # mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体：解决plot不能显示中文问题
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    # ====================================  构建数据：2个类别  ========================================================
    X,y = make_classification(n_samples=100,n_features=3,n_redundant=0,n_classes=2,n_informative=1,
                              n_clusters_per_class=1,class_sep=0.5,random_state=0)
    # 将特征和类别拼接一起，类别在最后一个维度上
    data = np.concatenate((X,y[:,None]),axis=1)

    # ========================================= 2个类别降维到2维  =====================================================
    model = LDA(k=2)
    new_data,_ = model.fit(data)
    print(new_data[:5])
    # 展示降维后的数据
    plt.title('2个类别的分类数据降维到2维后的结果')
    plt.scatter(new_data[:,0].tolist(),new_data[:,1].tolist(),marker='o',c=new_data[:,2].tolist())
    plt.show()

    # ====================================  构建数据：3个类别  ========================================================
    X,y = make_classification(n_samples=200,n_features=3,n_redundant=0,n_classes=3,n_informative=2,
                              n_clusters_per_class=1,class_sep=0.5,random_state=0)
    # 将特征和类别拼接一起，类别在最后一个维度上
    data = np.concatenate((X,y[:,None]),axis=1)

    # ======================================== 多个类别降维到2维  ======================================================
    model = LDA(k=2)
    new_data,_ = model.fit(data)
    print(new_data[:5])
    # 展示降维后的数据
    plt.title('3个类别的分类数据降维到2维后的结果')
    plt.scatter(new_data[:,0].tolist(),new_data[:,1].tolist(),marker='o',c=new_data[:,2].tolist())
    plt.show()