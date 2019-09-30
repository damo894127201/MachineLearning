# -*- coding: utf-8 -*-
# @Time    : 2019/9/28 20:55
# @Author  : Weiyang
# @File    : MaxEntropy.py

#===============================================================================================================
# 最大熵模型(MaxEntropy)：概率模型学习的一个准则

# 含义：在满足约束条件的模型集合中，选取熵最大的模型P(Y|X)。

# 解释：最大熵原理认为，要选择的概率模型首先必须满足已有的事实，即约束条件；在没有更多信息的情况下，那些不确定的
#       部分都是“等可能的”。最大熵原理通过熵的最大化来表示等可能，熵是一个数值指标！！！

# 投资角度：我们平时谈到的不要把鸡蛋放在一个篮子里，从而降低风险；从信息论的角度，就是保留了最大的不确定性，即熵最大
#           当我们需要对一个事件的概率分布进行预测时，最大熵原理告诉我们所有的预测应当满足全部已知的条件，而对未知
#           的情况不要做任何主观假设(这点很重要)，也就是让概率分布最均匀，保留全部的不确定性，将预测的风险降到最小。

# 形式化定义：假设满足所有约束条件的模型集合为C，定义在条件概率分布P(Y|X)上的条件熵为
#                               H(P) = -∑{x,y}P^(x)*P(y|x)logP(y|x)
#             则模型集合C中，使条件熵H(P)最大的模型P(Y|X)，称为最大熵模型，式中的对数为自然对数。

# 最大熵模型P(Y|X)：
#                              P_w(y|x) = 1 / Z_w(x) * exp(∑w_i * f_i(x,y))
#                              Z_w(x) = ∑_y exp(∑w_i * f_i(x,y))
#        Z_w(x)为规范化因子，f_i(x,y)是特征函数；w_i是特征函数的权值；w 是w_i构成的参数向量。

# 最大熵模型学习的目标：在给定的训练数据的条件下，对模型进行极大似然估计或正则化的极大似然估计，即以似然函数为目标函数的最优化问题
#                       通常通过迭代算法求解，可以使用改进的迭代尺度算法(improved iterative scalling,IIS)、梯度下降、牛顿法或拟牛顿法。
#                       求解使似然函数极大的参数向量w

# 有几点注意：
# 1. 似然函数的写法：条件概率分布p(y|x)的对数似然函数为：
#                          L_p(P_w) = log∏_{x,y} P(y|x)^{P'(x,y)} = ∑_{x,y}P'(x,y)logP(y|x)
#    这与我们常见的似然函数的形式不同：
#                          L(x1,x2,x3,..xn;θ) = log∏P(xi;θ)
#    实际上是一样的，只不过是似然函数的另一种写法。我们只需要在第一个似然函数前乘以数据个数N，就可以发现
#     L_p(P_w) =  ∑_{x,y} N * P'(x,y)logP(y|x) =  ∑_{x,y} n_{x,y} *logP(y|x) = log∏_{x,y} P(y|x)^n_{x,y}
#     其中，n_{x,y}表示取某类情况的数据个数；我们写似然函数时，自然要把相同情况下以及不同情况下的概率相乘，把同类的融合在一起
#     写成指数形式，就是如上情况。

# 2. 特征函数：一般说的特征都是指输入的特征，而最大熵模型中的“特征”指的是输入和输出共同决定的特征
#    1. 仅对输入X抽取特征，即特征函数f(X)；
#    2. 最大熵模型同时对输入X和输出Y抽取特征，即特征函数f(X,Y),这里X是特征向量中的某一个分量；
#    3. 每个特征函数都有一个权重w_i;
#    4. 特征函数f(x,y)描述输入x和输出y之间的某一个事实，这个事实是自己定义的，f(x,y)具体为：
#                                   1 , x与y满足某一事实
#                         f(x,y) =
#                                   0，否则
#       例如，x = x1,y = y1 为某一事实，则f(x=x1,y=y1) = 1，类似于词频；也可以定义为更复杂些的，
#       x_1 = x1,x_2 = x2,x_3 = x2,y = y1 为某一事实，x_1表示特征1，x_2表示特征2，.....

# 3. 改进的迭代尺度算法IIS
#    1. 基本思想：假设最大熵模型当前的参数向量是W=(w1,w2,...,wn)^T,我们希望找到一个新的参数向量
#                            W+δ=（w1+δ1,w2+δ2,w3+δ3,....,wn+δn）
#                 使得模型的对数似然函数值增大。如果能有这样一种参数向量更新的方法：W -> W+δ,那么就可以重复使用这一方法，
#                 直至找到对数似然函数的最大值。
#    2. 思路：L(W+δ) - L(W)    >=    A(δ|W)(利用不等式-logα>= 1 - α做变换得到的结果)    >= B(δ|W) (利用Jensen不等式做变换)
#             B(δ|W)对δ_i求偏导，再令偏导取0，可得到δ_i的取值方式；
#             由于δ是一个向量，含有多个变量，不易同时优化。IIS试图一次只优化其中一个变量δ_i，而固定其它变量 δ_j,j≠i
#   3. 对所有特征函数的权重w_i，初值取0，然后逐个更新，直至所有的权重w_i收敛

# 4. 适用于离散型特征，对于连续型特征，得二值化，三值化，。。将其转为离散型特征
# 5. 缺点：参数多，计算量大，工程实现的方法决定了模型实用与否

# 与朴素贝叶斯模型比较：最大熵模型的目标是 寻找使条件熵 H(P) 最大的模型P(Y|X)，然后依据P(Y|X)，做预测；
#                      朴素贝叶斯模型的目标是利用先验概率P(Y)和P(X|Y)，找到使得后验概率P(Y|X)最大的类别Y，作为预测的类别；
# 与逻辑斯蒂回归LR的关系：LR模型和最大熵模型具有类似的形式，都属于对数线性模型；如果最大熵模型的特征函数只考虑Y而不考虑X
#                        且Y取值只有0,1两种，特征函数满足
#                                                   f(x,y=1) = 1
#                                                   f(x,y=0) = 0
#                        此时最大熵模型就是二项LR模型。
#===============================================================================================================

import numpy as np
from collections import defaultdict

class MaxEntropy:
    '''
    最大熵模型：用于离散型特征，IIS算法实现

    特征函数的定义：
    这里我们定义 f(x=xi,y=yi) = 1，即在训练数据中，只要出现过的特征对(xi,yi),就有f(x=xi,y=yi) = 1
    因此，在对各个类别的概率进行预测时，如果数据中出现(x=xj,y=yi)特征对，则f(x=xj,y=yi)=0；
    但有一点需要特别注意，因为各个特征分量xi的取值可能很相似，因此我们务必要对特征分量xi进行编码，
    具体做法是 对每个特征分量xi的取值，我们用如下形式来表示特征对(str(feature_index)+':'+str(xi),yi)
    这样来达到区分各个特征取值的目的。
    '''
    def __init__(self,X,Y,lr=None,max_iter=4000,threshold=1e-2):
        '''
        X = [[feature1,feature2,...],...] ，X最好不要使用numpy数组，因为编码后的特征无法完全存储在numpy数组内
        Y = [label,...]
        注: X 是训练数据的特征向量，各个特征分量是离散型随机变量，且X中可以存在缺失值，但缺失值需用None来表示
        因此在预测时，如果预测数据中存在缺失值，同样可以进行预测，就是有多少信息利用多少信息。
        '''
        # 确保X是python列表
        if type(X) is np.ndarray:
            self.X = X.tolist()
        else:
            self.X = X
        self.Y = Y
        self.n_samples = len(X) # 样本数
        self.n_features = len(X[0]) # 特征数
        self.feature_function = [] # [(xi,yi),...] 存储具体的特征函数，即只要特征对(xi,yi)在特征函数列表里，特征函数值为1，否则为0
        self.feature_functionID = defaultdict(int) # 存储特征函数的索引ID
        self.featurePairNumber = defaultdict(int) # key 为 (xi,yi),value 为 count(xi,yi) ，用于计算经验分布 P(X,Y)
        self.W = []  # 存储各个特征函数的权重,各个特征函数的权重初始值为0
        self.prior_W = []  # 存储上一轮特征函数的权重向量，各个特征函数的权重初始值为0
        self.pxy_ep = []  # 存储特征函数f_i(x,y)关于经验分布P(X,Y)的期望值
        self.model_px_ep = []  # 存储特征函数f_i(x,y)关于模型P(Y|X)和经验分布P(X)的期望值
        self.labels = list(set(Y)) # 训练数据中的y类别
        self.M = lr # 学习率，本质上是所有特征函数数量的倒数，不过我们可以将其当做学习率来对待
        self.numFeatureFunction = 0 # 特征函数的数量；如果未定义学习率，我们会用此值的倒数，替代学习率
        self.max_iter = max_iter # 最大迭代次数
        self.threshold = threshold # 用于判断权重向量W各个分量是否收敛的阈值

    def _EncodeX_and_initParams(self):
        '''
        对特征分量编码: str(feature_index) + ':' + str(feature) ；
        统计特征函数的数量，以及每个特征对的数量 ；
        '''
        # 遍历每条数据
        for sample_index in range(self.n_samples):
            # 开始编码，如果某个特征分量为None，则不编码
            for feature_index in range(self.n_features):
                if self.X[sample_index][feature_index] != None:
                    new_feature = str(feature_index) + ':' + str(self.X[sample_index][feature_index]) # 编码后的特征
                    self.X[sample_index][feature_index] = new_feature
                    # 特征函数
                    pair = (new_feature,self.Y[sample_index])
                    # 特征函数列表，用于判断数据中是否存在我们定义的特征函数取值对，若有，特征函数值取1；否则取0
                    self.feature_function.append(pair)
                    # 统计每个特征对的数量,用于计算经验分布P(X,Y)
                    self.featurePairNumber[pair] += 1
        # 对特征函数列表去重
        self.feature_function = list(set(self.feature_function))
        # 构建特征函数的索引
        for ID,pair in enumerate(self.feature_function):
            self.feature_functionID[pair] = ID
        # 特征函数的数量
        self.numFeatureFunction = len(self.feature_function)
        # 初始化每个特征函数的权重
        self.W = [0.0] * self.numFeatureFunction
        self.prior_W = self.W[:]
        # 计算每个类别的特征函数f_i(x,y)关于经验分布P(X,Y)的期望值
        self._pxy_ep()

    def _pxy_ep(self):
        '''
        计算特征函数f(x,y)关于经验分布P(X,Y)的期望值
        '''
        # 初始化特征函数f_i(x,y)关于经验分布P(X,Y)的期望值
        self.pxy_ep = [0.0] * self.numFeatureFunction
        # 逐一计算每个特征函数对应的期望,实际上是遍历各个特征分量及其可能的取值
        for index in range(self.numFeatureFunction):
            # 计算当前特征函数对应的经验分布p(xi,yi),
            # 然后与特征函数f_i(xi,yi)=1相乘，再求和，便得到期望
            pair = self.feature_function[index]
            ep = self.featurePairNumber[pair] / self.n_samples * 1.0
            self.pxy_ep[index] = ep

    def _calculate_pyx(self,X):
        '''
        根据输入X，计算条件概率P(Y|X)，返回一个字典,形式为{label:prob,..}
        X = [feature1,...]
        '''
        Zw = 0.0 # 规范化因子
        pyx = {} # 保存预测的各个类别的概率
        pyx_numerator = {} # 保存pyx的分子
        for label in self.labels:
            sum = 0.0
            # 逐一计算各个特征函数
            for feature in X:
                # 如果特征为空
                if feature == None:
                    continue
                if (feature,label) in self.feature_function:
                    sum += self.W[self.feature_functionID[(feature,label)]] * 1.0 # 指数
            numerator = np.exp(sum) # 分子
            pyx_numerator[label] = numerator
            Zw += numerator
        for label in self.labels:
            pyx[label] = pyx_numerator[label] / Zw
        return pyx

    def _model_px_ep(self):
        '''
        计算特征函数f(x,y)关于模型P(Y|X)和经验分布P(X)的期望值，具体而言我们要计算每个类别的特征函数相应的期望
        这里特征函数的类别是指，当特征分量取第一个特征时，为一类特征函数；取第二个类别时，为一类特征函数
        '''
        # 计算特征函数f(x,y)关于模型P(Y|X)和经验分布P(X)的期望值时，按理来说应该先计算p(x),再计算p(y|x)，但是我们的特征函数
        # 定义的是每一个特征分量与y之间的关系，p(x)容易计算，但是我们无法直接计算p(y|x)，因为只有一个特征分量；
        # 因此，我们通过计算每个样例所有分量预测而得的对每个类的概率，来计算p(y|x),即我们认为当前样本预测到的p(y)可以作为
        # p(y|x)的值，而p(x)的值，在累加过程已经计算了
        # 逐一计算每个训练样本的对每个类别的预测概率

        # 初始化特征函数f_i(x,y)关于模型P(Y|X)和经验分布P(X)的期望值
        self.model_px_ep = [0.0] * self.numFeatureFunction
        for sample in self.X:
            pyx = self._calculate_pyx(sample)
            px = 1.0 / self.n_samples
            for label in self.labels:
                for index,feature in enumerate(sample):
                    if (feature,label) in self.feature_function:
                        ID = self.feature_functionID[(feature,label)]
                        self.model_px_ep[ID] += px * pyx[label]

    def train(self):
        '''训练模型'''
        # 初始化参数
        self._EncodeX_and_initParams()

        # 如果未定义学习率
        if self.M == None:
            self.M = 1.0 / self.n_features
        # 开始训练
        for i in range(1,self.max_iter+1):
            self.prior_W = self.W[:] # 保存上一轮权重值
            self._model_px_ep() # 更新特征函数f(x,y)关于模型P(Y|X)和经验分布P(X)的期望值
            # 逐个更新每个权重wi的值
            for j in range(len(self.W)):
                self.W[j] += self.M * np.log(self.pxy_ep[j] / self.model_px_ep[j])
            print("Iter:%d..." % i,self.W)
            # 判断是否收敛
            if self._isConvergence(self.W,self.prior_W) and i > 0.5 * self.max_iter:
                break

    def _isConvergence(self,W1,W2):
        '''用于判断是否所有的权重wi都收敛'''
        for k in range(len(W1)):
            if np.abs(W1[k] - W2[k]) > self.threshold:
                return False
        return True

    def predict(self,X):
        '''
        对数据进行预测
        X = np.array([[feature1,...],...])
        '''
        results = []
        all_results = []
        for sample in X :
            # 对没有编码过的特征进行编码
            encode_features = []
            for index in range(len(sample)):
                if sample[index] == None:
                    encode_features.append(None)
                    continue
                encode_feature = str(index) + ':' + str(sample[index])
                encode_features.append(encode_feature)
            sample = encode_features
            prob = self._calculate_pyx(sample)
            # 选取类别较大的类作为预测结果
            prob = sorted(prob.items(),key=lambda x:x[1],reverse=True)
            results.append(prob[0][0])
            all_results.append(prob)
        return np.array(results),all_results # 返回预测的类别，返回预测的各个类别的概率