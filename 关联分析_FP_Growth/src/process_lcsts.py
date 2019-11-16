# -*- coding: utf-8 -*-
# @Time    : 2019/11/16 12:16
# @Author  : Weiyang
# @File    : process_lcsts.py

#=======================================================================================================================
# 对LCSTS2.0中的PART_III数据做处理
# 1. 分词
# 2. 去除停用词
# 3. 去除非中文字符 (可选)
# 4. 去重单字字符
# 5. 去重文章中的字符

# 可以文章为单位时，也可以将将文章拆分成句子，以句子为单位进行处理；这里我们以文章为单位

# 输入原始数据 ../data/Article_III.txt ，停用词 ../data/中文停用词表.txt
# 输出：文章序号,单词序列 ；其中单词以','分隔
#=======================================================================================================================

import jieba as jb
import pandas as pd
import re

stopwords = []
with open('../data/百度停用词表.txt','r',encoding='utf-8') as fi:
    for line in fi:
        stopwords.append(line.strip())
articles = {} # 存储文章，即key是文章编号，value是单词序列
count = 0
with open('../data/Article_III.txt','r',encoding='utf-8') as fi:
    for doc in fi:
        words = [word for word in jb.cut(doc.strip()) if '\u4e00' <= word <= '\u9fff']  # 剔除非中文字符
        #words = [word for word in jb.cut(doc.strip())]
        words = [word for word in words if word not in stopwords] # 去除停用词
        words = [word for word in words if len(word) > 1] # 去除单字字符
        words = list(set(words)) # 去重
        words.sort()
        word_sequence = ','.join(words)
        if len(word_sequence.strip()) == 0:
            continue
        articles[count] = word_sequence
        count += 1
# 输出
df = pd.DataFrame(columns=('文章编号', '单词序列'))
for id in range(len(articles)):
    df.loc[id] = [id+1,articles[id]] # 文章编号，单词序列
df.to_excel('../data/Articles.xlsx',encoding='utf-8',index=False,header=True)