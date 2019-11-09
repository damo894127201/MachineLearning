# -*- coding: utf-8 -*-
# @Time    : 2019/11/9 0:20
# @Author  : Weiyang
# @File    : examples_plot.py

#=======================================================================================================================
# 对文档进行潜在语义分析
# 选取k个主题，将单词和文档都投影到潜在语义空间内，即用向量表示单词和文档，向量的每一维度都是主题的取值
# 潜在语义分析后 ：
#                                 X_m*n ≈ T_m*k * Y_k*n
# T_m*k 的每一行表示一个单词，即单词对应的用主题表示的向量
# (Y_k*n)^T 的每一行表示一篇文档，即文档对应的用主题表示的向量

# 然后将两者在平面或三维中展示出来，然后进行高斯聚类
#=======================================================================================================================

from LSA import LSA
import matplotlib.pyplot as plt
from pylab import mpl
import numpy as np

mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体：解决plot不能显示中文问题
#mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


model = LSA(filePath='../data/documents.txt')
W, H = model.fit(model.matrix_frequent.values, n_topics=2, model='NMF')
H = H.T # 注意使用H时，需要转置一下

plt.scatter(W.values[:,0],W.values[:,1],color='b',marker='v')
plt.scatter(H.values[:,0],H.values[:,1],color='r',marker='o')
# 给点加上图例，分别表示单词和文档
plt.legend(["单词","文档"])
# Axis labels
plt.suptitle("单词 和 文档 在潜在语义空间(话题向量空间)的 投影")
plt.xlabel('主题 1')
plt.ylabel('主题 2')
plt.show()

id = np.random.randint(0,len(W.index),10) # 随机显示一些单词
for word in W.index[id]:
    points1 = plt.scatter(W.loc[word].values[0], W.loc[word].values[1], color='b',marker='v')
    plt.text(W.loc[word].values[0],W.loc[word].values[1] + 0.02,word,fontsize='xx-small') # 给点加标签

id = np.random.randint(0,len(H.index),10) # 随机显示一些文档
for doc in H.index[id]:
    points2 = plt.scatter(H.loc[doc].values[0], H.loc[doc].values[1], color='r',marker='o')
    plt.text(H.loc[doc].values[0], H.loc[doc].values[1] + 0.02,doc,fontsize='xx-small')

# points1  和 points2 用于记录单词和文档的最后一个点，便于展示图例
plt.legend([points1,points2],["单词","文档"])
# Axis labels
plt.suptitle("单词 和 文档 在潜在语义空间(话题向量空间)的 投影")
plt.xlabel('主题 1')
plt.ylabel('主题 2')
plt.show()