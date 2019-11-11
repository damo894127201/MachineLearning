# -*- coding: utf-8 -*-
# @Time    : 2019/11/10 10:14
# @Author  : Weiyang
# @File    : PLSA_slow.py

#======================================================================================================================
# 概率潜在语义分析(probabilistic latent semantic analysis,PLSA)：含有隐变量的模型，学习算法是EM算法
# 也称概率潜在语义索引(probabilistic latent semantic indexing,PLSI),是一种利用概率生成模型对文本集合进行话题分析的无监督学习
# 方法。模型最大的特点是 用隐变量表示话题；整个模型表示 文本生成话题，话题生成单词，从而得到 单词-文本共现数据 的过程；假设每个
# 文本由一个话题分布决定，每个话题由一个单词分布决定。
#
# 概率潜在语义分析受潜在语义分析的启发提出，两者可以通过矩阵分解关联起来。概率潜在语义分析模型中的矩阵U'和V'是非负的、规范化的
# 表示条件概率分布，而潜在语义分析模型中的矩阵U和V是正交的，未必非负，并不表示概率分布。

# 概率潜在语义分析就是发现由隐变量表示的话题，即潜在语义。

# 概率潜在语义分析模型：生成模型 和 共现模型
# 1. 生成模型：
#    1. 概念：生成模型表示文本生成话题，话题生成单词，从而得到 单词-文本共现数据T 的过程；假设每个文本由一个话题分布决定，
#             每个话题由 一个单词分布决定。
#             单词变量 w 和 文本变量 d 是观测变量，话题变量z 是 隐变量，生成模型的定义如下：
#             单词-文本共现数据T，是一个矩阵，其行表示单词，列表示文本，元素表示 单词-文本对(w,d)的出现次数。单词-文本共现数据T
#             的生成概率为所有单词-文本对(w,d)的生成概率的乘积，即
#                                  P(T) = ∏_(w,d) P(w,d)^n(w,d)
#             每个单词-文本对(w,d)的生成概率为：
#                                  P(w,d) = P(d)P(w|d) = P(d) * (∑_{z} P(z|d) * P(w|z))
#              P(d) 是文档出现的概率，可以直接统计出来；P(z|d) 是 文本d 生成话题z 的条件概率分布；
#              P(w|z)是话题z 生成 单词w 的条件概率分布；n(w,d)表示(w,d)的出现次数
#    2. 属概率有向图模型
#    3. 生成模型假设在话题z给定条件下，单词w与文本d条件独立，即
#                                 P(w,z|d) = p(z|d)P(w|z,d) = p(z|d)P(w|z)
# 2. 共现模型：
#    1. 概念：生成模型描述 单词-文本共现数据T 拥有的模式，共现模型的定义如下：
#                                 P(T) = ∏_(w,d) P(w,d)^n(w,d)
#                                 P(w,d) = ∑_z P(z)P(w|z)P(d|z)
#    2. 属概率有向图模型。
#    3. 共现模型假设在话题z给定条件下，单词w 与 文本d 是条件独立的，即
#                                 P(w,d|z) = P(w|z)P(d|z)
# 3. 生成模型与共现模型在概率公式意义上是等价的，但是拥有不同的性质。生成模型刻画单词-文本共现数据T 生成的过程，共现模型描述
#    单词-文本共现数据T 拥有的模式。
# 4. 学习策略：观测数据的极大似然估计
# 5. 学习算法：EM算法，EM算法是一种迭代算法，每次迭代包括交替的两步：E步，求期望；M步，求极大。E步是计算Q函数，即完全数据的对
#              数似然函数对不完全数据的条件分布的期望。M步是对Q函数求极大化，更新模型参数，这一步一般采用拉格朗日法，求得参数
#              的解析解。
#    1. 概率潜在语义分析模型是含有隐变量的模型，目标函数对数似然函数的优化无法用解析方法求解，故用EM算法。
#    2. E步：计算Q函数，Q函数为完全数据的对数似然函数对不完全数据的条件分布的期望
#    3. M步：极大化Q函数，一般是通过约束最优化求解Q函数，采用拉格朗日法求解参数的解析解
#    4. 算法执行过程：
#       1. 输入：单词集合W={w1,w2,...,w_M},文本集合D={d1,d2,...,d_N}，话题集合Z={z1,z2,...,z_K},共现数据{n(wi,dj)}，其中
#                i=1,2,...,M ; j=1,2,...,N
#       2. 输出：话题z_k 生成 单词w_i 的条件概率分布P(w_i|z_k) 和 文本d_j 生成话题z_k 的条件概率分布P(z_k|d_j)
#       3. 迭代执行以下E步，M步，直到收敛为止
#          E步：简化后的Q函数：
#                                P(z_k | w_i,d_j) = P(w_i|z_k)P(z_k|d_j) / {∑_{k=1,2,..,K} P(w_i|z_k)P(z_k|d_j)}
#          M步：
#                 P(w_i|z_k) =
#                            ∑_{j=1,..,N}n(w_i,d_j)P(z_k|w_i,d_j) / {∑_{m=1,..M}∑_{j=1,..,N}n(w_m,d_j)P(z_k|w_m,d_j)}
#
#                 P(z_k|d_j) =
#                            ∑_{i=1,...,M}n(w_i,d_j)P(z_k|w_i,d_j) / n(d_j)
#                 n(d_j) = ∑_{i=1,...,M}n(w_i,d_j) ，表示文本d_j中的单词个数
#                 n(w_i,d_j)表示单词w_i在文本d_j中出现的次数
# 6. 模型参数：如果直接定义单词与文本的共现概率P(w,d)，模型参数的个数是O(M*N)，其中M是单词数，N是文本数。概率潜在语义分析的
#              生成模型和共现模型的参数个数是O(M*K + N*K) ，其中K是话题数。现实中，K<<M，所以概率潜在语义分析通过话题对数据
#              进行了更简洁地表示，减少了学习过程中过拟合的可能性。
#======================================================================================================================

import numpy as np
import jieba as jb
from collections import defaultdict
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')

class PLSA(object):
    '''概率潜在语义分析(概率潜在语义索引)：生成模型的 EM算法实现'''

    def __init__(self,filePath=None):
        self.matrix_frequent,self.token2id,self.id2token,self.doc2id,\
        self.id2doc,self.wordDict,self.num_Documents = self._loadData(filePath)

    def _loadData(self,filePath):
        '''读取文档集数据，返回 两个 单词-文档矩阵X ，其中每个元素分别是是 词频 和 TF-IDF
        filePath ：文档集路径，其中，每一行代表一篇文档'''
        wordDict = [] # 单词词典
        Documents = [] # [[{word:num},{word:num},...],...] ，存储每篇文档中每个词的词频
        with open(filePath,'r',encoding='utf-8') as fi:
            for doc in fi:
                words = [word for word in jb.cut(doc.strip()) if '\u4e00' <= word <= '\u9fff'] # 剔除非中文字符
                # 统计每篇文档中每个词的词频
                docContent = defaultdict(int)
                for word in words:
                    docContent[word] += 1
                Documents.append(docContent)
                wordDict.extend(words) # 单词加入词包
        # 对词包去重
        wordDict = list(set(wordDict))
        # 对单词和文档进行编码
        token2id = {token:id for id,token in enumerate(wordDict)}
        id2token = {id:token for id,token in enumerate(wordDict)}
        doc2id = {'Doc: '+ str(id):id for id in range(len(Documents))}
        id2doc = {id:'Doc: '+ str(id) for id in range(len(Documents))}

        # 单词向量空间模型：单词-文档矩阵T(单词-文本共现数据T)
        matrix_frequent = np.zeros((len(wordDict),len(Documents)))
        for i,word in enumerate(wordDict):
            for j in range(len(Documents)):
                matrix_frequent[i][j] = Documents[j][word] * 1.0 # 元素为词频
        matrix_frequent = pd.DataFrame(matrix_frequent,columns=['Doc: '+ str(id) for id in range(len(Documents))],
                                       index=wordDict)
        return matrix_frequent,token2id,id2token,doc2id,id2doc,wordDict,len(Documents)

    def fit(self,X,n_topics=5,max_iters=300,threshold=1e-5):
        '''概率潜在语义分析
        X 是单词-文本共现数据T,pd.DataFrame结构
        n_topics 人为设定的主题的个数
        max_iters 最大的迭代次数
        threshold 前后两次参数的差值，小于该阈值，则停止迭代

        输入：
            共现数据{n(w_i,d_j)}，即单词-文本共现数据T，每行代表一个单词，每列代表一个文本，每个元素存储的都是单词在相应文档中的词频
        输出：
            话题z_k 生成 单词w_i 的条件概率分布P(w_i|z_k) 和 文本d_j 生成话题z_k 的条件概率分布P(z_k|d_j) ，这是两个矩阵
        '''
        logger = logging.getLogger('Training')

        # 初始化 话题z_k生成单词w_i的条件概率分布P(w_i|z_k),每一行代表一个主题，每一列代表一个单词，
        # 每个元素表示相应的行主题生成对应列单词的概率，同一个主题生成所有单词的概率之和为1，即每一行元素的和为1
        topic_generate_word_matrix = np.ones((n_topics,len(self.wordDict))) / float(len(self.wordDict))
        # 转为pd.DataFrame
        topic_generate_word_matrix = pd.DataFrame(topic_generate_word_matrix,index=['Topic:' + str(i) for i in range(n_topics)],
                                                  columns=self.wordDict)

        # 初始化 文本d_j 生成话题z_k 的条件概率分布P(z_k|d_j)，每一行代表一个文档，每一列代表一个主题，
        # 每个元素表示相应的行文档生成对应列单词的概率，同一个文档生成所有主题的概率之和为1，即每一行元素的和为1
        doc_generate_topic_matrix = np.ones((self.num_Documents,n_topics)) / float(n_topics)
        # 转为pd.DataFrame
        doc_generate_topic_matrix = pd.DataFrame(doc_generate_topic_matrix,index=['Doc: ' + str(id) for id in range(self.num_Documents)],
                                                 columns=['Topic:' + str(i) for i in range(n_topics)])

        # Q函数 P(z_k|w_i,d_j)
        word_and_doc_generate_topic_matrix = np.zeros((len(self.wordDict) * self.num_Documents, n_topics))
        # 转为pd.DataFrame
        index = ['(' + word + ',' + 'Doc: '+ str(id) + ')' for word in self.wordDict for id in range(self.num_Documents)]
        word_and_doc_generate_topic_matrix = pd.DataFrame(word_and_doc_generate_topic_matrix,
                                                          index=index,
                                                          columns=['Topic:' + str(i) for i in range(n_topics)])
        # 主题
        Topics = ['Topic:' + str(i) for i in range(n_topics)]

        for step in range(1,max_iters+1):
            # 拷贝一份参数矩阵，用于比较前后更新之间参数变化的大小
            doc_generate_topic_matrix_copy = np.copy(doc_generate_topic_matrix.values)
            topic_generate_word_matrix_copy = np.copy(topic_generate_word_matrix.values)

            # E步：求期望，即Q函数 P(z_k|w_i,d_j)
            # 先遍历每个主题
            for topic in Topics:
                # 存储 同一主题下的概率之和，即分母
                sum_value = 0
                # 遍历每个单词
                for word in self.wordDict:
                    # 遍历每个文档
                    for doc in doc_generate_topic_matrix.index:
                        temp = topic_generate_word_matrix.loc[topic][word] * doc_generate_topic_matrix.loc[doc][topic]
                        sum_value += temp
                        word_and_doc_generate_topic_matrix.loc['(' + word + ',' + doc + ')'][topic] = temp
                # 该列除以分母
                if sum_value != 0:
                    word_and_doc_generate_topic_matrix[topic] /= float(sum_value)

            # M步：求Q函数的极大，更新参数
            # 先遍历每个主题
            for topic in Topics:
                # 记录P(w_i|z_k)的分母的值
                sum_value1 = 0
                # 遍历每个单词
                for word in self.wordDict:
                    # 记录P(w_i|z_k)的分子的值
                    sum_value2 = 0
                    # 遍历每个文档
                    for doc in doc_generate_topic_matrix.index:
                        temp = X.loc[word][doc] * word_and_doc_generate_topic_matrix.loc['(' + word + ',' + doc + ')'][topic]
                        sum_value2 += temp
                    # 更新P(w_i|z_k)参数
                    topic_generate_word_matrix.loc[topic][word] = sum_value2
                    sum_value1 += sum_value2
                # 更新P(w_i|z_k)参数
                if sum_value1 != 0:
                    topic_generate_word_matrix.loc[topic] /= float(sum_value1)

                # 更新P(z_k|d_j)
                # 遍历每个文档
                for doc in doc_generate_topic_matrix.index:
                    # 记录P(z_k|d_j)分母的值
                    sum_value = 0
                    # 遍历每个单词
                    for word in self.wordDict:
                        temp = X.loc[word][doc] * word_and_doc_generate_topic_matrix.loc['(' + word + ',' + doc + ')'][topic]
                        sum_value += temp
                    # 更新参数
                    doc_generate_topic_matrix.loc[doc][topic] = sum_value / float(np.sum(X[doc]))
            # 比较前后两次迭代参数差值之和是否达到指定阈值，即参数是否稳定
            # 差值计算
            difference1 = np.sum(np.sum(np.abs(topic_generate_word_matrix.values - topic_generate_word_matrix_copy),axis=0))
            difference2 = np.sum(np.sum(np.abs(doc_generate_topic_matrix.values - doc_generate_topic_matrix_copy),axis=0))
            difference = difference1 + difference2
            logger.info('epochs:{}\tdifferences:{}'.format(step, difference))
            if difference < threshold:
                logger.info('Training Finished!')
                break
        return topic_generate_word_matrix,doc_generate_topic_matrix

if __name__ == '__main__':
    model = PLSA(filePath='../data/documents.txt')
    W,H = model.fit(model.matrix_frequent,n_topics=5)
    print(W)
    print(H)







































