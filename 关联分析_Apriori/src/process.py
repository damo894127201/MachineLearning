# -*- coding: utf-8 -*-
# @Time    : 2019/11/12 22:32
# @Author  : Weiyang
# @File    : process.py

# =====================================================================================================================
# ../data/data.xlsx中，有两列：销售单明细 和 商品编码(脱过敏)，其中每一行表示一条交易和一件商品
# 本模块的目的在于
# 将../data/data.xlsx数据转为 每一行表示一条交易事务，和该交易对应的所有商品序列，商品间以 , 隔开
# =====================================================================================================================

import pandas as pd
from collections import defaultdict

data = pd.read_excel('../data/data.xlsx',index=False)
transaction = defaultdict(set)
for id in data.index:
    transaction[data.loc[id]['销售单明细']].add(data.loc[id]['商品编码'])
new_data = pd.DataFrame(index=range(len(transaction.keys())),columns=['销售单编号','商品编码序列'])
for id,key in enumerate(transaction.keys()):
    new_data.loc[id]['销售单编号'] = key
    new_data.loc[id]['商品编码序列'] = ','.join(list(transaction[key]))
# 写入到Excel表中,index表示是否需要行号，header表示是否需要列名等头部
new_data.to_excel('../data/transaction.xlsx',encoding='utf-8',index=False,header=True)