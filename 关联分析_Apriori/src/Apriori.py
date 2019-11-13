# -*- coding: utf-8 -*-
# @Time    : 2019/11/12 10:29
# @Author  : Weiyang
# @File    : Apriori.py

#=======================================================================================================================
# 关联规则分析：Apriori算法实现
# 目标是：发现 频繁项集 和 关联规则，首先找到频繁项集，然后才能获取关联规则；

# 关联分析(Association Analysis)或关联规则学习(Association Rule Learning)：无监督学习
# 关联规则分析也称为购物篮分析，最早是为了发现超市销售数据库中不同商品之间的关联关系，可以用于回答“哪些商品经常被同时购买”。
# 关联规则是反映一个事物与其他事物之间的关联性，若多个事物之间存在着某种关联关系，那么其中的一个事物就能通过其他事物预测到。
# 举例：
#       菜品的合理搭配是有规律可循的。顾客的饮食习惯、菜品的荤素口味，有些菜品是相互关联的，而有些菜品之间是对立或竞争关系
#       (负关联)，这些规律都隐藏在大量的历史菜单数据中。如果能够通过数据挖掘发现客户点餐的规则，就可以快速识别客户的口味，当
#       他下了某个菜品的订单时推荐相关的菜品，引导客户消费，提高客户的就餐体验和餐饮企业的业绩水平。

#       通过查看哪些商品经常在一起购买，可以帮助商店了解用户的购买行为。这种从数据海洋中抽取的知识可以用于商品定价、市场促销
#       存货管理等环节；

#       在美国国会投票记录中发现关联规则。在一个国会投票记录的数据集中发现议案的相关性，使用分析结果来为政治竞选活动服务，或者
#       预测选举官员会如何投票；

#       在Twitter源中发现一些共现词，对于给定的搜索词，发现推文中频繁出现的单词集合

#       从新闻网站点击流中挖掘新闻流行趋势，挖掘哪些新闻广泛被用户浏览到

#       搜索引擎推荐，在用户输入查询词时，推荐相关的查询词项

#       啤酒与尿布：据报道，美国中西部的一家连锁店发现，男人们会在周四购买尿布和啤酒，这样商店实际上可以将尿布与啤酒放在一起
#                   并确保在周四全价销售从而获利，当然，这家商店并没有这么做。

# 关联分析是在大规模数据集中寻找有趣关系的任务，这些关系可以有两种形式：
# 1. 频繁项集(Frequent Item Sets)：是经常出现在一起的物品的集合，例如{葡萄酒，尿布，豆奶}就是频繁项集的一个例子；
#    项集(ItemSet)：包含0个或多个项的集合，如果包含K个项，则称为K-项集；
#    事务：比如商场客户的一条交易记录，包含了很多种商品
#    事务的宽度：事务中出现的项的个数
# 2. 关联规则(Association Rules)：暗示两种物品之间可能存在很强的关系，例如{尿布}->{葡萄酒}的关联规则，这意味着如果有人买了尿布
#                                 那么他很有可能也会买葡萄酒；注意，{葡萄酒}->{尿布}是另一条规则；
#                                 箭头左边的集合称作前件，箭头右边的集合称作后件；
# 3. 支持度(support)：用来量化项集的频繁程度，定义为数据集中包含该项集的记录所占的比例，例如有5条交易记录，其中有3条包含
#                    {豆奶，尿布}，因此{豆奶，尿布}的支持度为 3/5 。
#                     支持度是针对项集来说的，因此可以定义一个最小的支持度，而只保留满足最小支持度的项集。
# 4. 可信度或置信度(confidence)：是针对一条诸如 {尿布}->{葡萄酒}的关联规则来定义的。这条规则的可信度被定义为
#                               “支持度({尿布，葡萄酒})/(支持度{尿布})”，例如{尿布，葡萄酒}的支持度为3/5，{尿布}的支持度
#                                为4/5，所以 {尿布}->{葡萄酒} 的可信度为3/4=0.75，这意味着对于包含“尿布”的所有记录，我们
#                                的规则对其中75%的记录都适用。

# 支持度和可信度是用来量化关联分析是否成功的方法。假设想找到支持度大于0.8的所有项集，应该如何去做？一个办法是生成一个物品所有
# 可能组合的清单，然后对每一种组合统计它出现的频繁程度，但当物品成千上万时，上述做法非常非常慢。Apriori算法会减少关联规则学习
# 时所需的计算量。

# 我们一般使用三个指标来度量一个关联规则的强度：支持度，置信度 和 提升度
# 1. 支持度(Support)：项集X,Y同时发生的概率称为关联规则 X->Y的支持度(也可以反过来，但是反过来后就是另一个规则了)，即
#                            Support(X->Y) = P(X∩Y) = P(XY) = X,Y同时发生的事件个数 / 总事件数
#                     支持度其实就是频率；
# 2. 置信度(Confidence)：项集X发生的前提下，项集Y发生的概率 称为 关联规则X->Y的置信度，即
#                            Confidence(X->Y) = P(Y|X) = X,Y同时发生的事件个数 / X发生的事件个数
#                                                      = support(X,Y同时发生) / support(X发生)
#                        置信度是一个条件概率
# 3. 提升度(Lift)：项集X发生的事件中包含项集Y的个数/项集X发生的事件个数 再除以 项集Y发生的事件个数/总事件数
#                  提升度是针对项集Y而言的，即在项集X发生的前提下，项集Y发生的可能性占项集Y总可能性的比例
#                  Lift(Y) = Confidence(X->Y) / P(Y) = (P(X∩Y)/P(X)) / P(Y) = P(Y|X) / P(Y)
#                                                                            =  Confidence(X->Y) / support(Y)
#                  提升度反映了关联规则中的X与Y的相关性，提升度>1 且越高表明X与Y正相关性越高；
#                  提升度<1且越低表明负相关性越高；提升度=1，表明X与Y没有相关性；
#                  例如，在没有任何条件下，项集Y出现的比例是0.75，而出现项集X且同时出现项集Y的比例是0.67，也就是说设置了项集X
#                  这个条件，项集Y出现的比例反而降低了，这说明项集X与项集Y是排斥的。

# 关联规则分析与朴素贝叶斯分类器的区别
# 朴素贝叶斯分类器和关联规则都用到先验知识，但是贝叶斯是多个概率推导True/False，关联规则是解决A->Who(关联规则)的问题

# 从大规模数据集中寻找物品间的隐含关系被称作 关联分析 或 关联规则学习 ，这里的主要问题在于，寻找物品的不同组合是一项十分耗时
# 的任务，所需的计算代价很高，蛮力搜索方法并不能解决这个问题，所以需要用更智能的方法在合理的时间范围内找到频繁项集，比如Apriori

# Apriori算法原理：目的是避免项集和规则数目的指数增长，从而在合理时间内计算出频繁项集
# 1. 如果某个项集是频繁的(支持度大于我们指定的阈值)，那么它的所有子集也是频繁的；
# 2. 如果一个项集是非频繁集，那么包含它的所有超集也是非频繁的；(最有用！)
# 3. 从1和2可推理：若X->Y是强规则，则X,Y,XY都必须是频繁项集；

# Apriori在拉丁语中指“来自以前”。当定义问题时，通常会使用先验知识或者假设，这被称作“一个先验”(a priori)。在贝叶斯统计中
# 使用先验知识作为条件进行推断很常见，先验知识可能来自领域知识、先前的一些测量结果等。

# Apriori算法流程：
# 1. 算法输入：标准化项集ItemSets,最小支持度minSupport,最小自信度minConfidence
# 2. 生成频繁项集：
#    1. 扫描数据集，对每个项进行计数，生成候选1-项集(元素只包含1个项)
#    2. 保留支持度小于阈值的项，生成频繁1-项集(元素只包含1个项)
#    3. 从2-项集开始，由频繁k-1项集 生成 频繁k-项集
#        1. 频繁k-1项集两两组合，判定是否可以连接，若能则连接成k-项集；连接的方式：若有两个k-1项集，每个项集保证有序，如果
#           两个k-1项集的前k-2个项相同，而最后一个项不同，则证明它们是可连接的，即可连接生成k-项集
#        2. 对k-项集中的每个项集检测其子集是否频繁，去除子集不是频繁项集的k-项集，即不在频繁k-1项集中的项集
#        3. 扫描数据集，计算k-项集的支持度，去除支持度小于阈值的项集，生成k-项集
#    4. 若当前k-项集中只有一个项集时循环结束
# 3. 从频繁项集中挖掘关联规则
#    1. 循环扫描频繁项集的每个项，针对每个频繁项做如下处理
#    2. 从当前频繁项开始，生成规则，并创建一个规则列表T，来存储规则，下面的方法也叫分级法：
#        1. 其中规则右部(后件)只包含一个元素(项)，规则左部包含剩余元素，然后对这些规则进行测试，即判断是否满足最小可信度，
#           将满足要求的规则加入到事先创建的规则列表T中，并保留一份拷贝，命名为S；
#        2. 对上轮生成的规则列表S，将其中规则右部(后件)的元素(项)个数增加1个，规则左部包含剩余元素，然后测试，将满足条件的规则
#           分别加入T中，和 替换S中的规则
#        3. 循环下去，直至S中的规则为空，或新生成的规则左部只剩下1个元素为止
#    3. 如果某条规则不满足最小可信度要求，那么该规则的所有子集也不会满足最小可信度要求，例如 假设规则0,1,2->3并不满足
#       最小可信度要求，那么就可以断定任何左部为{0,1,2}子集的规则也不会满足最小可信度要求，这样便可以大量减少需要判断
#       是否满足可信度要求的规则数目；
# 4. 输出频繁项集和规则，以及其支持度，置信度和提升度

# 算法输入：
# 1. 最小支持度和最小置信度
# 2. 商品交易事务的数据集：每一行表示一条交易事务，和该交易对应的所有商品序列，商品间以 , 隔开，即
#    销售单编号            商品编码序列
#    1317060100203          2BPJ006,4DCS002
#    1317060600067          2BYF034,2BKB006,2BPB031
#    注：在商品编码序列中的商品不重复
# 3. 算法输出
#    频繁项(Items)    推荐项(Recommend)   支持度(support)   置信度(confidence)   提升度(lift)
#    注：频繁项->推荐项 表示 一条关联规则，频繁项是前件，推荐项是后件
#=======================================================================================================================

import pandas as pd
from collections import defaultdict
import copy
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')

class Apriori(object):
    '''Apriori算法实现关联分析'''
    def __init__(self,minSupport,minConfidence):
        self.minSupport = minSupport # 最小支持度
        self.minConfidence = minConfidence # 最小置信度

    def _loadData(self,filePath):
        '''加载数据'''
        data = pd.read_excel(filePath,index=False)
        return data

    def isConnection(self,items1,items2):
        '''判断频繁项集items1和items2是否可以连接
        连接的方式：若有两个k-1项集，每个项集保证有序，如果两个k-1项集的前k-2个项相同，而最后一个项不同，
        则证明它们是可连接的，即可连接生成k-项集

        items1 和 items2的格式为字符串，如下形式：
        "item,item,item,..,item"
        '''
        items1 = items1.split(',')
        items2 = items2.split(',')
        if items1[:-1] == items2[:-1]:
            if items1[-1] != items2[-1]:
                return True
        return False

    def isExist(self,newItem,transaction):
        '''判断新的候选频繁项是否存在于某条事务中
        newItem: 'item,item,..,item' 是一个字符串
        transaction: 'item,item,..,item' 是一个字符串
        '''
        newItems = newItem.split(',')
        transaction = transaction.split(',')
        # 返回属于集合newItems的元素但不属于集合transaction的元素，
        # 如果结果不含任何元素，则newItems包含于transaction中
        result = set(newItems).difference(transaction)
        if len(result) == 0:
            return True
        return False

    def generateFrequentItem(self,data):
        '''生成频繁项集,data是pd.DataFrame'''
        logger = logging.getLogger('GenerateFrequentItem')
        itemSets = []  # 存储所有不重复的单个项
        itemSupport = defaultdict(int)  # 存储每个项的支持度
        transaction = {} # 存储事务，key是事务编号，value是商品编码序列，并且升序排过的字符串
        col1,col2 = data.columns

        # 生成候选1-项集
        for id in data.index:
            items = data.loc[id][col2].split(',')
            items.sort() # 升序排序
            transaction[id] = ','.join(items)
            itemSets.extend(items)
            for item in items:
                itemSupport[item] += 1 / len(data.index)
        # 生成所有不重复的项，并按照首字符升序排列
        itemSets = list(set(itemSets))
        logger.info('共有{}种不同的单项(商品)'.format(len(itemSets)))
        itemSets.sort()
        count = 1
        # 生成频繁1-项集
        for item in itemSets:
            # 如果支持度小于指定的阈值，则删除
            if itemSupport[item] < self.minSupport:
                itemSupport.pop(item)
            logger.info('new_FrequentItem: {}'.format(item))
            logger.info('第{}种候选项集'.format(count))
            count += 1

        # 如果一个项集是非频繁的，那么包含它的所有超集也是非频繁的，因此频繁项集的项数 最多是 频繁1-项集的数目，即k值最大
        # 是 频繁1-项集的数目，实际上k值的取值取决于，k-1项频繁项集生成的k项频繁项集的数目，如果k项频繁项集的数目为1
        # 则停止生成k+1项频繁项集，因为k项频繁项集的生成是基于k-1项集，且必须保证其所有子集都是频繁项集
        one_FrequentItems = list(itemSupport.keys()) # 1项频繁项集
        k_FrequentItems = copy.deepcopy(one_FrequentItems) # k-项 频繁项集
        # 由k-项 频繁项集 生成 k+1 -项 频繁项集，从2-项频繁项集开始生成
        while True:
            # 判断k-项 频繁项集的个数是否为1
            if len(k_FrequentItems) <= 1:
                break
            # k+1项频繁项集
            k_1_FrequentItems = []
            # k-项 频繁项集 两两组合生成k+1项频繁项集
            for i in range(len(k_FrequentItems)-1):
                # 获取第一个频繁项的各项
                items1 = k_FrequentItems[i].split(',')
                for j in range(i+1,len(k_FrequentItems)):
                    # 判断k_FrequentItems[i]和k_FrequentItems[j]是否可以连接
                    if self.isConnection(k_FrequentItems[i],k_FrequentItems[j]):
                        items2 = k_FrequentItems[j].split(',')
                        # 生成k+1项候选频繁集，为避免出现'a,b'和'b,a'之类的频繁项集，我们需要统一将频繁项集升序排序
                        new_items = items1 + items2
                        new_items = list(set(new_items)) # 去重
                        new_items.sort()
                        new_items = ','.join(new_items)
                        # 计算k+1项候选频繁集的支持度
                        support = 0
                        for key in transaction:
                            # 判断该k+1项候选频繁项集是否存在，即其各项是否同时存在于某条事务中
                            if self.isExist(new_items,transaction[key]):
                                support += 1 / len(data.index)
                        if support >= self.minSupport:
                            k_1_FrequentItems.append(new_items) # 将new_items添加到k+1项频繁项集中
                            itemSupport[new_items] = support # 记录new_items的支持度
                        logger.info('new_FrequentItem: {}'.format(new_items))
                        logger.info('第{}种候选项集'.format(count))
                        count += 1
            # 更新k-项 频繁项集
            k_FrequentItems = k_1_FrequentItems[:]
        # 返回频繁项子集及其支持度
        return itemSupport

    def generateRule(self,itemSupport):
        '''生成关联规则'''
        logger = logging.getLogger('GenerateRule')
        ruleSupport = {} # 存储每个规则支持度
        ruleConfidence = {} # 存储每个规则的置信度
        ruleLift = {} # 存储每个规则的提升度

        count = 1 # 记录共生成过多少个规则

        # 循环扫描频繁项集的每个项
        for items in itemSupport.keys():
            # 分裂出当前频繁项集的每个项
            items = items.split(',')
            # 如果频繁项只有一个项，则无法生成规则，跳出当前循环
            if len(items) <= 1:
                continue

            k_item_rules = [] # 存储当前频繁项生成的规则,即右件含有k项的规则，初始时右件还有1个项

            # 开始生成规则,规则的形式如下：{items(左件)}->{items(右件)}，
            # 右件的item个数从1开始增加到n-1，n表示该频繁项的项个数

            # 生成右件只有1项的候选关联规则
            for i in range(len(items)):
                remaining_items = list(set(items) - set([items[i]])) # 除去当前的项余下的项
                remaining_items.sort() # 排序
                rule = ','.join(remaining_items) + '->' + items[i] # 生成规则
                # 判断规则的置信度是否大于最小置信度
                confidence = itemSupport[','.join(items)] / float(itemSupport[','.join(remaining_items)])
                if confidence >= self.minConfidence:
                    ruleConfidence[rule] = confidence
                    k_item_rules.append(rule)
                    # 计算该规则的支持度
                    ruleSupport[rule] = itemSupport[','.join(items)]
                    # 计算该规则的提升度,即该规则对规则右件的提升度
                    lift = confidence / float(itemSupport[items[i]])
                    ruleLift[rule] = lift
                    logger.info('new rule: {}'.format(rule))
                logger.info('第{}种候选项关联规则,其置信度为：{}'.format(count,confidence))
                count += 1

            # 基于右件有k项的关联规则，生成右件有k+1项的关联规则,此时新规则的左件减少一项，依据是：
            # 如果某条规则不满足最小可信度要求，那么该规则的所有子集也不会满足最小可信度要求，例如 假设规则0,1,2->3并不满足
            # 最小可信度要求，那么就可以断定任何左部为{0,1,2}子集的规则也不会满足最小可信度要求，这样便可以大量减少需要判断
            # 是否满足可信度要求的规则数目
            while True:
                # 如果仅剩余一个规则，则不再合并，退出循环
                if len(k_item_rules) <= 1:
                    break
                k_1_item_rules = [] # 用于存储右件是k+1项的新规则
                # 将规则两两合并，合并的依据是：将两个规则左件相同的项保留作为新规则的左件，将两个规则右件的不同项合并且去重
                # 作为新规则的右件；当两个规则左件没有相同的项时，则不进行合并；
                for i in range(len(k_item_rules)):
                    # 获取第一个规则的各项
                    left_items_1,right_items_1 = k_item_rules[i].split('->')
                    left_items_1 = left_items_1.split(',')
                    right_items_1 = right_items_1.split(',')
                    for j in range(i+1,len(k_item_rules)):
                        # 获取第二个规则的各项
                        left_items_2, right_items_2 = k_item_rules[j].split('->')
                        left_items_2 = left_items_2.split(',')
                        right_items_2 = right_items_2.split(',')
                        # 判断两个规则是否可以合并，即比较两个规则的左件是否有相同的项，求交集
                        common_items = list(set(left_items_1) & set(left_items_2))
                        if len(common_items) > 0:
                            # 对两个规则的右件进行合并，即求合并
                            merge_items = list(set(right_items_1) | set(right_items_2))
                            # 生成新规则
                            common_items.sort() # 先排序，避免出现相同的规则
                            merge_items.sort()
                            rule = ','.join(common_items) + '->' + ','.join(merge_items)
                            # 判断规则是否存在
                            if rule in ruleConfidence:
                                continue
                            # 计算新规则的置信度
                            confidence = itemSupport[','.join(items)] / float(itemSupport[','.join(common_items)])
                            # 判断新规则是否满足最小置信度
                            if confidence >= self.minConfidence:
                                ruleConfidence[rule] = confidence
                                k_1_item_rules.append(rule)
                                # 计算该规则的支持度
                                ruleSupport[rule] = itemSupport[','.join(items)]
                                # 计算该规则的提升度,即该规则对规则右件的提升度
                                lift = confidence / float(itemSupport[','.join(merge_items)])
                                ruleLift[rule] = lift
                                logger.info('new rule: {}'.format(rule))
                            logger.info('第{}种候选项关联规则,其置信度为：{}'.format(count,confidence))
                            count += 1
                # 判断前后两次之间规则是否有变化，若无变化则退出循环
                k_item_rules.sort() # 排序，便于比较
                k_1_item_rules.sort()
                if k_item_rules == k_1_item_rules:
                    break
                # 更新规则右件是k个项的规则集合
                k_item_rules = k_1_item_rules[:]
        # 返回规则的支持度，置信度，提升度
        return ruleSupport,ruleConfidence,ruleLift

    def fit(self,filePath):
        '''生成频繁项集和关联规则'''
        data = self._loadData(filePath) # 加载数据
        itemSupport = self.generateFrequentItem(data) # 生成频繁项集和事务集合
        ruleSupport, ruleConfidence, ruleLift = self.generateRule(itemSupport) # 生成规则的支持度，置信度，提升度
        return ruleSupport, ruleConfidence, ruleLift,itemSupport


if __name__ == '__main__':
    model = Apriori(minSupport=0.03,minConfidence=0.5)
    ruleSupport, ruleConfidence, ruleLift ,itemSupport = model.fit(filePath='../data/transaction2.xlsx')
    # 输出
    df1 = pd.DataFrame(columns=('规则', '支持度', '置信度', '规则对规则右件的提升度')) # 规则
    # 规则的另一种表达方式，即规则的前件作为Item，规则的后件作为Recommend
    df2 = pd.DataFrame(columns=('Item','Recommend','Support','Confidence','Lift'))
    rules = list(ruleConfidence.keys()) # 获取规则
    rules.sort()
    for id,rule in enumerate(rules):
        df1.loc[id] = [rule,ruleSupport[rule],ruleConfidence[rule],ruleLift[rule]]
        left_item,right_item = rule.split('->')
        df2.loc[id] = [left_item,right_item, ruleSupport[rule], ruleConfidence[rule], ruleLift[rule]]
    # 输出到同一个Excel的不同sheet表
    with pd.ExcelWriter('../data/result2.xlsx') as writer:
        df1.to_excel(writer,encoding='utf-8',index=False,sheet_name='规则表')
        df2.to_excel(writer,encoding='utf-8',index=False,sheet_name='推荐项目及权重')