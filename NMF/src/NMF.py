# -*- coding: utf-8 -*-
# @Time    : 2019/11/8 10:28
# @Author  : Weiyang
# @File    : NMF.py

#=====================================================================================================================
# 非负矩阵分解(Non-negative Matrix Factorization)：用于话题分析，数据压缩，矩阵分解的一种
# 1.非负矩阵的概念：若一个矩阵的所有元素非负，则称该矩阵为非负矩阵，若X是非负矩阵，则记作 X >=0
# 2.非负矩阵分解：给定一个非负矩阵X >=0 ，找到两个非负矩阵 W>=0 和 H>=0，使得
#                                           X ≈ W * H
#                 即将非负矩阵X 分解为两个非负矩阵W和H的乘积的形式，称为非负矩阵分解。因为 WH 与 X 完全相等很难实现，所以只
#                 求 WH 与 X 近似相等。W称为基矩阵，H为系数矩阵。

# 3.假设非负矩阵X是 m*n 矩阵，非负矩阵W和H分别为 m*k矩阵 和 k*n矩阵。假设 k < min(m,n) ，即 W和H 小于原矩阵X，所以非负矩阵分解
#   是对原数据的压缩。
# 4.非负矩阵分解的形式化：可以形式化为最优化问题求解
#    1. 定义损失函数或代价函数：
#       1. 平方损失函数：||A - B||^2 = ∑_{i,j}(a_ij - b_ij)^2  , a_ij 和 b_ij 分别是矩阵A和B的元素，下界为0，当A=B时取到
#       2. 散度：D(A||B) = ∑_{i,j}(a_ij * log(a_ij / b_ij) - a_ij + b_ij)，a_ij 和 b_ij 分别是矩阵A和B的元素，下界为0，
#                当A=B时取到
#    2. 转化为如下最优化问题：
#       1. 目标函数||X - WH||^2 关于 W 和 H的最小化，满足约束条件W,H >= 0,即
#                                            min_{W,H}  ||X - WH||^2
#                                            s.t.        W,H >= 0
#       2. 目标函数D(X||WH) 关于 W 和 H的最小化，满足约束条件W,H >= 0,即
#                                            min_{W,H}  D(X||WH)
#                                            s.t.        W,H >= 0
#       3. 由于目标函数||X - WH||^2 和 D(X||WH) 只是对变量 W 和 H 之一的凸函数，而不是同时对两个变量的凸函数，因此找到全局
#          最优(最小值)比较困难，可以通过数值最优化方法求局部最优。
#          1. 梯度下降法：容易实现，但是收敛速度慢
#          2. 共轭梯度法：收敛速度快，但实现比较复杂
#          3. 基于乘法更新规则的优化算法：交替对 W 和 H 进行更新，该方法本质上是梯度下降法，通过定义特殊的步长和非负的初始值，
#                                        保证迭代过程及结果的矩阵W和H均为非负。其理论依据是如下的定理：
#             1.定理1：平方损失||X - WH||^2对下列乘法更新规则
#                                 H_lj  <-- H_lj * (W^T * X)_lj / (W^T * W * H)_lj  (式1.1)
#                                 W_il  <-- W_il * (X * H^T)_il / (W * H * H^T)_il  (式1.2)
#               其中，i=1,..,m
#                     l=1,...,k
#                     j=1,...,n
#                     X_m*n ,W_m*k ,H_k*n
#               是非增的。当且仅当W和H是平方损失函数的稳定点时，函数的更新不变。
#             2.定理2：散度损失D(X||WH)对下列乘法更新规则
#                                 H_lj  <-- H_lj * (∑_i [W_il * X_ij / (W * H)_ij]) / (∑_i W_il)
#                                 W_il  <-- W_il * (∑_j [H_lj * X_ij / (W * H)_ij]) / (∑_i H_lj)
#               其中，i=1,..,m
#                     l=1,...,k
#                     j=1,...,n
#                     X_m*n ,W_m*k ,H_k*n
#               是非增的。当且仅当W和H是散度损失函数的稳定点时，函数的更新不变。
# 5.非负矩阵分解的迭代算法：基于乘法更新规则
#   1. 输入：单词-文本矩阵 X_m*n >=0，文本集合的话题个数k，最大迭代次数max_steps
#   2. 输出：话题矩阵W_m*k 和 文本矩阵 H_k*n
#   3. 具体步骤：
#      1. 初始化：
#         W >= 0,并对W的每一列数据归一化
#         H >= 0
#      2. 迭代
#         迭代次数由1到max_steps执行下列步骤
#         1. 更新W的元素，对 l 从1到k，i从1到m 按式1.1 更新W_il
#         2. 对W的每一列数据归一化
#         3. 更新H的元素，对 l 从1到k，j从1到n 按式1.2 更新H_lj
#=====================================================================================================================

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')

def NMF(X,k=5,max_steps=200,threshold=1e-7):
    '''非负矩阵分解的迭代算法
    X 是一个非负矩阵，形式为np.ndarray
    k 是话题的个数 或 特征数
    max_steps 是最大的迭代次数
    threshold 是相邻迭代之间||X - WH||^2的值，当差值小于threshold则停止迭代'''
    X = X if type(X) == np.ndarray else np.array(X)
    m,n = X.shape
    # 初始化话题矩阵W 和 文本矩阵H
    W = np.ones((m,k))
    W = W / np.sqrt(m) # 对W的列向量归一化
    H = np.ones((k,n))

    loss = 10000 # 记录损失

    logger = logging.getLogger('NMF')

    # 迭代更新
    for step in range(1,max_steps+1):
        # 更新W
        X_H_T = X.dot(H.T)
        W_H_H_T = (W.dot(H)).dot(H.T)
        W = W * X_H_T / W_H_H_T

        # 对W 的列进行归一化，使得每个列是一个单位向量
        W = W / np.sqrt(np.sum(W*W,axis=0))

        # 更新H
        W_T_X = (W.T).dot(X)
        W_T_W_H = (W.T).dot(W).dot(H)
        H = H * W_T_X / W_T_W_H

        # 计算误差损失
        loss = X - W.dot(H)
        loss = np.sum(np.sum(loss * loss ,axis=0))
        logger.info('epochs:{}\tloss{}'.format(step, loss))

        if loss <= threshold:
            logger.info('Training Finished!')
            return W,H
    logger.info('Training Finished!')
    return W, H

if __name__ == '__main__':
    X = np.array([[1, 1],
                  [2, 1],
                  [3, 1.2],
                  [4, 1],
                  [5, 0.8],
                  [6, 1]])
    W,H = NMF(X,k=2)
    print('W matrix: ',W)
    print('H matrix: ',H)
    print()
    print('原矩阵是：',X)
    print()
    print('WH的结果：',W.dot(H))