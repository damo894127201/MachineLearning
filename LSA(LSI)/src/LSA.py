# -*- coding: utf-8 -*-
# @Time    : 2019/11/8 18:35
# @Author  : Weiyang
# @File    : LSA.py

#=====================================================================================================================
# 潜在语义分析(latent semantic analysis,LSA)：非概率的话题分析模型，无监督学习
# 是一种无监督学习方法，主要用于文本的话题分析，其特点是通过矩阵分解发现文本与单词之间的基于话题的语义关系。由于潜在语义分析
# 最初应用于文本信息检索，所以也被称为 潜在语义索引(latent semantic indexing,LSI),在推荐系统、图像处理、生物信息学等领域有
# 广泛应用。

# 文本信息处理中，传统的方法以单词向量表示文本的语义内容，以单词向量空间的度量 表示 文本之间的相似度。
# 潜在语义分析旨在解决这种方法不能准确表示语义的问题，试图从大量的文本数据中发现潜在的话题，以话题向量表示文本的语义内容，以
# 话题向量空间的度量更准确地表示文本之间的语义相似度。这也是话题分析(Topic modeling)的基本想法。

# 文本信息检索的任务：
# 任务是：当用户提出查询时，帮助用户找到与查询最相关的文本，以排序的形式展示给用户。
# 一个最简单的做法是 采用单词向量空间模型，将查询与文本分别表示为单词的向量，计算查询向量与文本向量的内积，作为语义相似度，以
#                   这个相似度的高低对文本进行排序。在这里，查询被看成是一个伪文本，查询与文本的语义相似度表示查询与文本的相关性。

# 1.向量空间模型(vector space model,VSM)
#   基本想法：给定一个文本，用一个向量表示该文本的“语义”，向量的每一维对应一个单词，其数值为该单词在该文本中出现的频数
#             或权值(TF-IDF)；
#   基本假设：文本中所有单词的出现情况表示了文本的语义内容；
#   语义相似度：文本集合中的每个文本都表示为一个向量，存在于一个向量空间中，向量之间的内积或标准化内积(内积再除以各自模长的乘积)
#               表示文本之间的“语义相似度”。两个文本的相似度并不是由一两个单词是否在两个文本中出现决定，而是由所有的单词在
#               两个文本中共同出现的“模式”决定。

# 2.单词向量空间模型(Word vector space model)
#   也叫 单词-文档矩阵(word-document matrix)，通过单词的向量表示文本的语义内容。
#   1. 定义：给定一个含有 n 个文本的集合D={d1,d2,...,dn}，以及在所有文本中出现的m个单词的集合W={w1,w2,...,wm}。将单词在文本
#            中出现的数据(频数或权值)用一个 单词-文本矩阵(word-document matrix)表示，并记作X
#                                      |x11,x12,...,x1n|
#                                      |x21,x22,...,x2n|
#                                X =   |... ... ... ...|
#                                      |xm1,xm2,...,xmn|
#            矩阵X的每一行表示一个单词，每一列表示一个文档，元素x_ij 表示单词wi在文档dj中出现的频数或权值。
#            由于单词的种类很多，而每个文本中出现单词的种类通常较少，所以 单词-文档矩阵 是 一个 稀疏矩阵。
#   2. 优点：模型简单，计算效率高。因为单词向量通常是稀疏的，两个向量的内积计算只需要在其同不为零的维度上进行即可，需要的计算
#            很少，可以高效地完成。
#   3. 局限性：内积相似度未必能够准确表达两个文本的语义相似度。因为自然语言的单词具有 一词多义性 及 多词一义性 ，即同一个单词
#              可以表示多个语义，多个单词可以表示同一个语义，所以基于单词向量的相似度计算存在不精确的问题。
#              无法解决 一词多义性 和 多词一义性

# 3.话题向量空间模型(Topic vector space model)
#   也叫 话题-文档矩阵(topic-document matrix)
#   1. 两个文本的语义相似度：可以 体现在 两者的 话题相似度上。一个文本一般含有若干个话题。如果两个文本的话题很相似，
#                           那么两者的语义应该也相似。给定一个文本，用话题空间的一个向量表示该文本，该向量的每一分量对应
#                           一个话题，其数值为该话题在该文本中出现的权值。
#      相似度的度量：用两个话题空间的向量的内积或标准化内积表示两个文本的语义相似度。
#
#   2. 话题(topic)：指文本所讨论的内容或主题。话题可以由若干个语义相关的单词表示，同义词可以表示同一个话题，而多义词可以表示
#                   不同的话题。
#   3. 基于话题的向量空间模型可以解决 基于单词的向量空间模型 存在的问题。
#   4. 定义：
#       1. 单词向量空间(单词-文档 矩阵)
#            给定一个文本集合D={d1,d2,...,dn} 和 一个相应的单词集合W={w1,w2,...,wm}。可以获得其 单词-文本矩阵X，X构成原始的
#            单词向量空间，每一列是一个文本在单词向量空间中的表示：
#                                      |x11,x12,...,x1n|
#                                      |x21,x22,...,x2n|
#                                X =   |... ... ... ...|
#                                      |xm1,xm2,...,xmn|
#            矩阵X 也可以写作 X = [x1,x2,...,xn]
#       2. 话题向量
#            假设所有文本共含有k个话题，假设每个话题由一个定义在单词集合W上的m维向量表示，称其为 话题向量，即
#                                t_j = [t1j,t2j,...,tmj]^T  ,j=1,2,...,k
#            这是一个列向量，其中tij是但从wi在话题tj中的权值，权值越大，该单词在该话题中的重要度就越高;
#            这 k 个话题向量t1,t2,...,tk 张成一个话题向量空间，维数为k。
#            注意 话题向量空间T 是单词向量空间X的一个子空间。
#       3. 话题向量空间(Topic vector space): 单词-话题 矩阵
#          话题向量空间T也可以表示为一个矩阵，称为 单词-话题 矩阵
#                                      |t11,t12,...,t1k|
#                                      |t21,t22,...,t2k|
#                                T =   |... ... ... ...|
#                                      |tm1,tm2,...,tmk|
#            矩阵T 也可以写作 T = [t1,x2,...,tk],每一行表示一个单词，每一列表示一个话题
#       4. 话题向量空间模型
#          定义：通过话题的向量表示文本的语义内容。假设有话题-文档 矩阵
#                                      |y11,y12,...,y1n|
#                                      |y21,y22,...,y2n|
#                                Y =   |... ... ... ...|
#                                      |yk1,yk2,...,ykn|
#          其中，每一行表示一个话题，每一列表示一个文本，每一个元素表示话题在文本中的权值。
#          话题向量空间模型认为，Y 的每一列向量是话题向量，表示一个文本，两个话题向量的内积或标准化内积表示文本之间的语义相似度。

# 4.单词-文档矩阵(word-document matrix) ： X，也叫单词向量空间模型

# 5.单词-话题矩阵(word-topic matrix): T

# 6.话题-文档矩阵(topic-document matrix): Y，也叫话题向量空间模型

# 7.潜在语义分析算法
#   1. 目的：潜在语义分析的目的是 找到合适的 “单词-话题 矩阵 T” 与 “话题-文本 矩阵Y”，将“单词-文本 矩阵X” 近似地表示为
#            T 与 Y 的乘积形式：
#                                  X_m*n ≈ T_m*k * Y_k*n
#            k 是话题的数量
#            等价地，潜在语义分析将文本在单词向量空间的表示 X ，通过线性变换T 转换为 话题向量空间中的表示Y。
#   2. 潜在语义分析的关键是对 单词-文本 矩阵X 进行以上的 矩阵因子分解(话题分析)。
#
#   3. 潜在语义分析将文本表示为单词-文本矩阵，然后对单词-文本矩阵进行分解，具体有两种方式：
#    1. 奇异值分解SVD
#       通过对 单词-文本 矩阵X进行截断奇异值分解，得到
#                                 X_m*n ≈ U_k*k ∑_k*k (V_k*k)^T =  U_k * (∑_k * (V_k)^T)
#       矩阵U_k 表示单词-话题 矩阵T(话题空间)，矩阵(∑_k * (V_k)^T) 表示 话题-文本 矩阵Y(文本在话题空间的表示)
#    2. 非负矩阵分解NMF
#       非负矩阵分解也可以用于话题分析，非负矩阵分解将非负的 单词-文本 矩阵 近似分解成 两个非负矩阵 W 和 H 的乘积，得到：
#                                 X_m*n ≈ W_m*k * H_k*n
#       矩阵W 表示单词-话题 矩阵T(话题空间)，矩阵H 表示 话题-文本 矩阵Y(文本在话题空间的表示)
#=====================================================================================================================

import numpy as np
from numpy import linalg as la
import jieba as jb
from sklearn.decomposition import NMF
from collections import defaultdict
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')

class LSA(object):
    '''潜在语义分析(潜在语义索引)：奇异值分解SVD实现 和 非负矩阵分解 实现'''

    def __init__(self,filePath=None):
        self.matrix_frequent,self.matrix_tfidf,self.token2id,self.id2token,self.doc2id,\
        self.id2doc,self.wordDict,self.num_Documents = self._loadData(filePath)

    def _loadData(self,filePath):
        '''读取文档集数据，返回 两个 单词-文档矩阵X ，其中每个元素分别是是 词频 和 TF-IDF
        filePath ：文档集路径，其中，每一行代表一篇文档'''
        wordDict = [] # 单词词典
        Documents = [] # [[{word:num},{word:num},...],...] ，存储每篇文档中每个词的词频
        wordInDocuments = defaultdict(int) # 存储每个单词在多少个文档中出现过
        with open(filePath,'r',encoding='utf-8') as fi:
            for doc in fi:
                words = [word for word in jb.cut(doc.strip()) if '\u4e00' <= word <= '\u9fff'] # 剔除非中文字符
                # 统计每篇文档中每个词的词频
                docContent = defaultdict(int)
                for word in words:
                    docContent[word] += 1
                Documents.append(docContent)
                # 统计每个单词在多少个文档中出现过，需要去重
                for word in set(words):
                    wordInDocuments[word] += 1
                wordDict.extend(words) # 单词加入词包
        # 对词包去重
        wordDict = list(set(wordDict))
        # 对单词和文档进行编码
        token2id = {token:id for id,token in enumerate(wordDict)}
        id2token = {id:token for id,token in enumerate(wordDict)}
        doc2id = {'Doc: '+ str(id):id for id in range(len(Documents))}
        id2doc = {id:'Doc: '+ str(id) for id in range(len(Documents))}

        # 单词向量空间模型：单词-文档 矩阵X
        matrix_frequent = np.zeros((len(wordDict),len(Documents)))
        matrix_tfidf = np.zeros((len(wordDict),len(Documents)))
        for i,word in enumerate(wordDict):
            for j in range(len(Documents)):
                matrix_frequent[i][j] = Documents[j][word] * 1.0 # 元素为词频
                # 元素为tfidf
                matrix_tfidf[i][j] = Documents[j][word] * np.log(float(len(Documents)) / (wordInDocuments[word]) + 1)
        matrix_frequent = pd.DataFrame(matrix_frequent,columns=['Doc: '+ str(id) for id in range(len(Documents))],
                                       index=wordDict)
        matrix_tfidf = pd.DataFrame(matrix_tfidf,columns=['Doc: '+ str(id) for id in range(len(Documents))],
                                       index=wordDict)
        return matrix_frequent,matrix_tfidf,token2id,id2token,doc2id,id2doc,wordDict,len(Documents)

    def fit(self,X,n_topics=5,model='NMF'):
        '''潜在语义分析，输出 单词-话题 矩阵W 和 话题-文档 矩阵H
        model = 'SVD' or 'NMF'
        '''
        logger = logging.getLogger('Training')
        if model == 'NMF':
            nmf = NMF(n_components=n_topics,max_iter=200)
            W = nmf.fit_transform(X)
            H = nmf.components_

            # 转为pd.DataFrame
            W = pd.DataFrame(W, index=self.wordDict, columns=['Topic:' + str(i) for i in range(n_topics)])
            H = pd.DataFrame(H, index=['Topic:' + str(i) for i in range(n_topics)],
                             columns=['Doc: ' + str(id) for id in range(self.num_Documents)])
            return W, H
        elif model == 'SVD':
            U,sigma,V_T = la.svd(X) # 注意这里的 sigma是奇异值的列表，不是对角阵
            # 获取矩阵X的秩，因为在截断奇异值分解中，我们只会保留前k个最大的奇异值对应的部分
            rank = np.linalg.matrix_rank(X)
            # 判断 要获取的主题个数是否大于待分解矩阵X的秩
            if n_topics >= rank:
                logger.info('SVD分解过程中，输入的主题个数大于待分解矩阵的秩，无法进行截断奇异值分解，'
                            '主题的个数 已自动处理为 矩阵的秩')
                # 截断奇异值分解
                U = U[:,:rank]
                sigma = np.diag(sigma[:rank]) # 将sigma转为对角阵
                V_T = V_T[:rank,:]

                # 转为pd.DataFrame
                W = pd.DataFrame(U,index=self.wordDict,columns=['Topic:' + str(i) for i in range(n_topics)])
                H = sigma.dot(V_T)
                H = pd.DataFrame(H,index=['Topic:' + str(i) for i in range(n_topics)],
                                 columns=['Doc: '+ str(id) for id in range(self.num_Documents)])
                return W,H
            else:
                # 截断奇异值分解
                U = U[:,:n_topics]
                sigma = np.diag(sigma[:n_topics])  # 将sigma转为对角阵
                V_T = V_T[:n_topics,:]

                # 转为pd.DataFrame
                W = pd.DataFrame(U,index=self.wordDict,columns=['Topic:' + str(i) for i in range(n_topics)])
                H = sigma.dot(V_T)
                H = pd.DataFrame(H,index=['Topic:' + str(i) for i in range(n_topics)],
                                 columns=['Doc: '+ str(id) for id in range(self.num_Documents)])
                return W,H

if __name__ == '__main__':
    model = LSA(filePath='../data/documents.txt')
    print(model.matrix_frequent)
    print(model.matrix_tfidf)
    W,H = model.fit(model.matrix_frequent.values,n_topics=5,model='NMF')
    print(W)
    print(H)