# -*- coding: utf-8 -*-
# @Time    : 2019/9/17 15:08
# @Author  : Weiyang
# @File    : XGBoostRegressionDecisionTree.py

#=======================================================================================================================
# XGBoostRegressionDecisionTree:XGBoost的基模型类
# XGBoost是GBDT的工程实现，因此其使用的基模型仍是回归树；GBDT使用的基模型是CART回归树
# XGBoost基模型相较于CART回归树差异体现在：
# 1. 分裂节点的准则：CART采用的是平方误差最小化准则；XGBoost基模型采用的是在损失函数的基础上
#                    加上正则项(叶子节点的个数+叶子结点取值的平方和)，通过将预测值(叶节点的值，由一阶导数和二阶导数可以计算出来)
#                    代入到损失函数中求得当前的最小损失函数值，再计算出分裂节点前后损失函数的差值，XGBoost通过最大化这个差值
#                    作为分裂节点的准则来构建决策树，即最大化分裂节点前后损失函数的差值来分裂节点，损失函数随着基模型的增加
#                     越来越小，直至小于某个阈值时，停止回归树的生成。
# 2. 损失函数：CART的损失函数是平方误差，XGBoost的损失函数可取(平方误差或逻辑损失，逻辑损失是由对数似然函数函数推导而得)
#              在用对数似然函数推导时，需要将标签y融合到概率中，即P(Y=y|w,x)=1 / (1 + exp(-yw.x))，然后再进行推导
#              推导后可知，逻辑损失本质上就是交叉熵
#              逻辑损失用于逻辑斯蒂回归，具体查看https://xgboost.readthedocs.io/en/latest/tutorials/model.html
# 3. 对损失函数的处理：对于平方误差损失函数MSE来说，损失函数对当前模型f(x_i)的一阶导数就是 负的残差(真实值-预测值)；
#                     对于其它损失函数来说，比如逻辑损失，并没有如此好的形式，因此XGBoost便对损失函数在f(x_(i-1))处做二阶泰勒展开，
#                     用G_i表示损失函数对f(x_(i-1))的一阶导数，H_i表示损失函数对f(x_(i-1))的二阶导数信息。
#                     在去除损失函数中的常数项后，损失函数变为 :
#                        sum(G_i * f(x_i) + 1/2 * H_i * f(x_i)^2) + 正则项
#                     我们令w
#                     这个损失函数便成为下一轮基模型,即新一棵决策树的学习目标。显然这个目标函数有一个优势，仅依赖于上一轮模型
#                      f(x_(i-1)) 一阶导数G_i 和二阶导数H_i 信息。这也是XGBoost之所以支持自定义损失函数的原因。
# 4. 叶节点的取值w(或叶节点的得分score)：CART回归树叶节点的取值为节点内样本数据的均值；
#                                      XGBoost各个叶节点的取值为w_j = - G_j / (H_j + lambda)，这个值通过对变形后的损失函数
#                                      对叶节点的得分w，求导数再令导数=0，可得。然后再代入损失函数中，便可得到目标损失函数的最小值。
# 5. 加入了正则项：正则项用来表示模型的复杂度 = gama * 叶节点的数量 + 1 / 2 * lambda * sum(每个叶节点取值的平方)
#                  防止过拟合，提高了模型的泛化能力
# 6. 剪枝策略：预剪枝，每次只优化一棵树(前向分步算法思想)，在分裂节点时，分裂节点前后的损失函数差值要大于一定的阈值，否则不分裂。
# 7. 模型的输入：GBDT的回归树(CART)每个模型的输入是(特征X,残差或负梯度),
#                而XGBoost的回归树的输入是(特征X，之前所有模型的输出值y_i的累加,真实值y)
#                GBDT的回归树中每个叶子节点的值，需要通过对 残差或负梯度取平均值得到；
#                XGBoost则不需要，它通过利用 (之前所有模型的输出值y_i的累加,真实值y)来求损失函数对上一个模型 一阶导数(梯度)和二阶导数
#                来求。也就是说GBDT用负梯度来拟合残差，来求得各个叶节点的取值；
#                XGBoost却不用拟合残差，直接计算出使损失函数最小的叶节点的取值(即回归树的预测值)
#=======================================================================================================================

import numpy as np
from DecisionTree import DecisionTree

class XGBoostRegressionDecisionTree(DecisionTree):
    '''XGBoost 模型的 基模型类，是一个回归树'''
    def _split(self,Y):
        '''
        由于XGBoost的基模型需要同时输入 y_true 和 y_pred
        因此，Y包含了 y_true 和 y_pred
        在计算叶子结点取值以及损失函数的最小值时，我们要用到y_true和y_pred
        来计算一阶导数和二阶导数信息，因此该函数的作用就是将两者分离成两个矩阵
        '''
        # 切割最后一个维度为两份
        num_dim = int(np.shape(Y)[1] / 2)
        y_true , y_pred = Y[:,:num_dim] , Y[:,num_dim:]
        return y_true , y_pred

    def _minLoss(self,y_true,y_pred,lambd):
        '''
        已知y_true和y_pred，通过求一阶导数和二阶导数，来求损失函数的最小值
        lambd 是损失函数中,正则化项 (所有叶节点值平方和sum(w^2)) 的系数
        '''
        # 计算当前叶子结点的G_j，需要将所有属于叶子节点j的样本的g_i累加
        G_j = np.power((self.loss.g(y_true,y_pred)).sum(),2)
        H_j = self.loss.h(y_true,y_pred).sum()
        # 该叶子节点的最小损失函数值
        minLoss = 0.5 *float(G_j) / (H_j + lambd)
        return minLoss

    def _lossReduction(self,y,y_l,y_r,lambd,gama):
        '''
        计算分裂节点前后损失函数的差值
        y 是 y_l 和 y_r 的并集,
        y_l是 y 拆分成左叶子结点的样本集，y_r是 y 拆分成右叶子节点的样本集
        lambd 是损失函数中,正则化项 (所有叶节点值平方和sum(w^2)) 的系数
        gama 是损失函数中，正则化项 叶节点个数|T| 的系数
        '''
        # 分裂开y_true和y_pred
        y = np.array(y)
        y_l = np.array(y_l)
        y_r = np.array(y_r)
        y,y_pred = self._split(y)
        y_l,y_l_pred = self._split(y_l)
        y_r,y_r_pred = self._split(y_r)

        # 分裂后左叶子节点的损失函数值
        left_loss = self._minLoss(y_l,y_l_pred,lambd)
        # 分裂后右叶子节点的损失函数值
        right_loss = self._minLoss(y_r,y_r_pred,lambd)
        # 分裂前节点的损失函数值
        all_loss = self._minLoss(y,y_pred,lambd)

        # 分裂前后损失函数的差值
        # 由于差值reduction必须为正，因此叶节点个数的系数gama还起到了剪枝的效果
        reduction = left_loss + right_loss - all_loss - gama

        return reduction

    def _approximate_update(self,y,lambd):
        '''
        计算叶节点的取值
        利用一阶导数和二阶导数计算叶节点的值
        lambd 是损失函数中,正则化项 (所有叶节点值平方和sum(w^2)) 的系数
        '''
        # 分裂y,得到y_true,y_pred
        y_true,y_pred = self._split(y)
        G_j = np.sum(self.loss.g(y_true,y_pred),axis=0)
        H_j =  np.sum(self.loss.h(y_true,y_pred),axis=0)
        leaf_value = - G_j / (H_j + lambd)
        return leaf_value

    def fit(self,X,Y):
        '''训练模型'''
        self.impurity_calculation = self._lossReduction
        self.leaf_value_calculation = self._approximate_update
        super(XGBoostRegressionDecisionTree,self).fit(X,Y)