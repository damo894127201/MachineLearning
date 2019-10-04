# -*- coding: utf-8 -*-
# @Time    : 2019/10/4 15:19
# @Author  : Weiyang
# @File    : util.py

#========================================================
# #在给定的位置和协方差画一个椭圆
#========================================================

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np

#给定的位置和协方差画一个椭圆
def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()
    #将协方差转换为主轴
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    #画出椭圆
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
#画图
def plot_gmm(X,labels,alphas,means,covariances, showLabel=True, ax=None):
    '''
    X 是聚类数据；
    labels是高斯混合模型预测的label；
    alphas是高斯混合模型各个分模型的权重或系数；
    means 是高斯混合模型各个特征分量的均值；
    covariances是高斯混合模型各个分模型的协方差矩阵
    showLabel:表示是否显示label
    '''
    ax = ax or plt.gca()
    if showLabel:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=4, cmap='viridis', marker='o',zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=4,  marker='o',zorder=2)
    ax.axis('equal')
    w_factor = 0.2 / alphas.max()
    for pos, covar, w in zip(means, covariances, alphas):
        draw_ellipse(pos, covar, alpha=w * w_factor)

def accuracy_score(y_true,y_pred):
    '''计算准确率'''
    accuracy = np.sum(y_true == y_pred,axis=0) / len(y_true)
    return accuracy