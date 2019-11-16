# -*- coding: utf-8 -*-
# @Time    : 2019/11/16 12:12
# @Author  : Weiyang
# @File    : FPGrowth_LCSTS.py

#=======================================================================================================================
# 基于FPGrowth_Rule.py模块抽取LCSTS2.O PART_III中的词对组合，需要事先对文本数据做如下处理：
# 1. 去除停用词
# 2. 去除非中文字符
# 3. 分词
#=======================================================================================================================

from FPGrowth_Rule import FPGrowth
import pandas as pd
import time


start = time.time()
fp = FPGrowth(minSupport=0.003, minConfidence=0.01)
ruleSupport, ruleConfidence, ruleLift, itemSupport = fp.fit(filePath='../data/Articles.xlsx')
# 输出
df1 = pd.DataFrame(columns=('规则', '支持度', '置信度', '规则对规则右件的提升度'))  # 规则
# 规则的另一种表达方式，即规则的前件作为Item，规则的后件作为Recommend
df2 = pd.DataFrame(columns=('Item', 'Recommend', 'Support', 'Confidence', 'Lift'))
# 存储频繁项集
df3 = pd.DataFrame(columns=('Item', 'Support')) # 存储所有的频繁项集
df4 = pd.DataFrame(columns=('Item', 'Support')) # 只存储项数大于1个的频繁项集

rules = list(ruleConfidence.keys())  # 获取规则
rules.sort()
for id, rule in enumerate(rules):
    df1.loc[id] = [rule, ruleSupport[rule], ruleConfidence[rule], ruleLift[rule]]
    left_item, right_item = rule.split('->')
    df2.loc[id] = [left_item, right_item, ruleSupport[rule], ruleConfidence[rule], ruleLift[rule]]
# 输出到同一个Excel的不同sheet表
with pd.ExcelWriter('../data/result3.xlsx') as writer:
    df1.to_excel(writer, encoding='utf-8', index=False, sheet_name='规则表')
    df2.to_excel(writer, encoding='utf-8', index=False, sheet_name='推荐项目及权重')
# 按照支持度从大到小排序频繁项集
items = [v[0] for v in sorted(itemSupport.items(),key=lambda p:p[1],reverse=True)]
# 输出频繁项集
count = 0
for id ,item in enumerate(items):
    # 存储所有的频繁项集
    df3.loc[id] = [item, itemSupport[item]]
    if len(item.split(',')) == 1:
        continue
    # 存储项数多于1个的频繁项集
    df4.loc[count] = [item, itemSupport[item]]
    count += 1
# 输出到同一个Excel的不同sheet表
with pd.ExcelWriter('../data/words_support.xlsx') as writer:
    df3.to_excel(writer, encoding='utf-8', index=False, sheet_name='所有的频繁项集')
    df4.to_excel(writer, encoding='utf-8', index=False, sheet_name='多个项的频繁项集')
#df3.to_excel('../data/words_support.xlsx',encoding='utf-8',index=False,header=True)
print('耗时：{}s'.format(time.time() - start))