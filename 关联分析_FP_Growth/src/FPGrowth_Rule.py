# -*- coding: utf-8 -*-
# @Time    : 2019/11/15 21:04
# @Author  : Weiyang
# @File    : FPGrowth_Rule.py

#======================================================================================================================
# FPGrowth算法抽取频繁项集和关联规则：
# 1. 利用FPGrowth算法抽取频繁项集和其支持度
# 2. 利用Apriori算法抽取关联规则
#======================================================================================================================

from FPTreeNode import FPTreeNode
import pandas as pd
from collections import defaultdict
import copy
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')

class FPGrowth(object):
    '''FPGrowth算法挖掘频繁项集'''
    def __init__(self,minSupport,minConfidence):
        self.minSupport = minSupport # 最小支持度
        self.minConfidence = minConfidence #最小置信度

    def _loadData(self,filePath):
        '''加载数据'''
        data = pd.read_excel(filePath,index=False)
        return data

    def createInitSet(self,data):
        '''对数据集处理,包裹为字典类型，并记录每条事务出现支持度'''
        transactionDict = defaultdict(int) # 存储数据集，key是事务，value是事务的支持度
        col1,col2 = data.columns
        numTransaction = len(data.index) # 数据总量
        for id in data.index:
            transaction = data.loc[id][col2].split(',')
            transaction.sort() # 排序一下，便于将相同的事务归为一处
            transactionDict[frozenset(transaction)] += 1 / numTransaction
        return transactionDict

    def createFPTree(self,transactionDict,minSupport):
        '''创建FP树
        transactionDict: 数据集，其中key是事务，是由各项构成的frozenset集合，value是该事务出现的支持度
        minSupport: 最小支持度
        '''
        headerTable = defaultdict(int) # 创建头指针表，key是每个项，value是该项的支持度
        # 第一次扫描数据集(不包含初始化数据集那一步)，逐个扫描每条事务
        for trans in transactionDict:
            # 扫描每个事务中的每个项
            for item in trans:
                # 记录每个项的支持度
                headerTable[item] += transactionDict[trans]
        # 扫描headerTable，删除支持度小于指定阈值的单一项，这里便利用了Apriori原理
        # 同时为头指针表的value增加一项指向相似结点的链接指针
        new_headerTable = {} # 用于保留支持度大于指定阈值的项
        for k in headerTable.keys():
            if headerTable[k] >= minSupport:
                # 保留支持度大于指定阈值的项，这些项便是1-项频繁集
                # 增加一项指向相似结点的链接指针
                new_headerTable[k] = [headerTable[k],None]
        headerTable = copy.deepcopy(new_headerTable) # 更新头指针表
        # 保存频繁项,此时保存的是1-项频繁集
        freqItemSet = set(headerTable.keys())
        # 如果1-项频繁集为空，则停止创建树
        if len(freqItemSet) == 0: return None,None # FP树，头指针表

        # 构建FP树
        # 创建根节点
        retTree = FPTreeNode('Null Set',1,None) # 结点名为'Null Set'，结点支持度为1，父节点为None
        # 第二次遍历数据集
        for trans,support in transactionDict.items():
            # 存储当前事务中支持度满足要求的项，便于更新树，或者说将该事务对应的前缀路径加入到树中
            localID = {}
            # 扫描每条事务的每个项
            for item in trans:
                if item in freqItemSet:
                    localID[item] = headerTable[item][0] # 记录每个项的支持度，便于对当前事务中的每个项进行排序
            # 如果存在满足要求的项，则将其添加进FP树中
            if len(localID) > 0:
                # 对事务中的每个项以支持度的大小进行排序，从大到小；如果每个项的支持度相同，则以项名的字母顺序表顺序进行排序
                # 因此这里排序有两个条件：第一个条件，先以项的支持度排序，从大到小；第二个条件，项名的字母表顺序，从小到大；
                # 由于第一个条件是决定项顺序的首要条件，因此必须放到最后进行最终的排序；
                # 这里排序，我们利用python sorted()排序的特性，即稳定性，也就是key相同的项，排序前后的相对次序是不变的；
                # 因此首先对项按字母顺序从小到大排序，当再对项按支持度排序时，只会调整那些支持度大于前项的项，支持度相同的项
                # 之间的相对顺序是不变的；因此，这样既能够保证各项之间以支持度大小逆序排序，也能够保证支持度相同的各项之间，
                # 按字母表顺序排序，使得在构造FP树时，遇到相同的事务(包含各项)时，能够输出一致的排序；
                # 最内层的排序，是对项名按字母表顺序从小到大排序；最外层的排序，是对各项以项的支持度从大到小逆序排序；
                # 下面两种方式都可以实现
                '''
                In [22]: b = {'a':1,'b':1,'ab':1,'bc':2,'d':3}
                In [23]: sorted(sorted(b.items(),key=lambda p:p[0]),key=lambda p:p[1],reverse=True)
                Out[23]: [('d', 3), ('bc', 2), ('a', 1), ('ab', 1), ('b', 1)]
                In [24]: sorted(b.items(),key=lambda p:(p[0],p[1]),reverse=True)
                Out[24]: [('d', 3), ('bc', 2), ('b', 1), ('ab', 1), ('a', 1)]
                '''
                # orderedItems = [v[0] for v in sorted(sorted(localID.items(), key=lambda p:p[0]),key=lambda p:p[1],reverse=True)]
                orderedItems = [v[0] for v in sorted(localID.items(), key=lambda p: (p[0], p[1]), reverse=True)]
                # 将该事务中的每个项添加进FP树中
                self.updateTree(orderedItems,retTree,headerTable,support)
        return retTree,headerTable # 返回FP树(即根节点)，头指针表

    def updateTree(self,items,FPTree,headerTable,support):
        '''将指定列表中的项，依次添加进指定的FP树中
        items: 按照支持度从大到小排序的项列表
        FPTree: 指定FP树的根节点
        headerTabel: 指定FP树的头指针表
        support: 该指定列表项的支持度
        '''
        # 先判断列表中第一个项是否是FP树根节点的直接子节点，若是则沿着该子节点添加后续的项
        if items[0] in FPTree.children:
            FPTree.children[items[0]].inc(support) # 如果存在，则增加该结点项的支持度
        else:
            # 如果不存在，则为FP树的根节点增加一个新的直接子节点，以存储该项
            FPTree.children[items[0]] = FPTreeNode(items[0],support,FPTree) # FPTree其实是根节点
            # 同时更新头指针表
            # 如果头指针表中该项还没有链接指针，则添加
            if headerTable[items[0]][1] == None:
                headerTable[items[0]][1] = FPTree.children[items[0]] # 将新创建的结点项作为头指针表中该项的链接指针
            else:
                # 若有链接指针，则将新创建的结点项添加到指针末尾
                self.updateHeader(headerTable[items[0]][1],FPTree.children[items[0]])
        # 如果列表项中还剩下有项，则递归添加到FP树中
        if len(items) > 1:
            # 注意这里传递进去的是前一个项对应的结点，这样可以避免再次从根节点开始遍历，同时也必须这么传
            self.updateTree(items[1:],FPTree.children[items[0]],headerTable,support)

    def updateHeader(self,firstNodeLink,targetNode):
        '''更新头指针表中的链接指针
        firstNodeLink: 头指针中某项的链接指针
        targetNode：待添加到链接指针末尾的结点项
        '''
        while(firstNodeLink.nodeLink != None):
            firstNodeLink = firstNodeLink.nodeLink
        # 将结点添加到链接指针的末尾
        firstNodeLink.nodeLink = targetNode

    def ascendTree(self,leafNode,prefixPath):
        '''从叶节点或中间节点回溯到根节点，输出该结点的前缀路径
        leafNode: 表示叶节点或中间节点，是起始结点
        prefixPath: 是一个列表，用于存储前缀路径中各项的名称，便于输出前缀路径
        '''
        if leafNode.parent != None:
            prefixPath.append(leafNode.name) # 将当前结点项的名称添加进前缀路径
            self.ascendTree(leafNode.parent,prefixPath)

    def findPrefixPath(self,treeNodeLink):
        '''寻找当前结点的所有前缀路径(条件模式基)
        treeNodeLink: 是头指针表中的每个项对应的链接指针
        '''
        # 用于存储当前项的所有条件模式基
        condPats = {} # key是条件模式基(前缀路径，不含当前项),value是其支持度
        while treeNodeLink != None:
            prefixPath = [] # 存储当前项的某个结点的前缀路径
            self.ascendTree(treeNodeLink,prefixPath)
            # 判断当前项的前缀路径中项的个数是否多于1个，因为前缀路径包含了当前项
            if len(prefixPath) > 1:
                condPats[frozenset(prefixPath[1:])] = treeNodeLink.support
            # 链接指针走到下一个相似的结点,即结点名称一样的结点
            treeNodeLink = treeNodeLink.nodeLink
        return condPats # 返回当前项的所有条件模式基(前缀路径)，都不含该项

    def _generateFreqItems(self,FPTree,headerTable,minSupport,prefixPath,freqItems):
        '''基于条件模式基生成条件FP树，进而生成频繁项集
        FPTree: FP树，即树的根节点，此处无用
        headerTable: 对应FP树的头指针表
        minSupport: 最小支持度
        prefixPath: 前缀路径，用来生成频繁项，也就是每段前缀路径对应了一个频繁项；随着前缀路径的扩大，频繁项也扩大；
        freqItems: 存储频繁项集及其支持度，是一个字典
        '''
        items = [v[0] for v in sorted(headerTable.items(),key=lambda p:p[1][0])] # 对头指针表中的项按支持度从小到大排序
        for item in items:
            backup_prefixPath = copy.deepcopy(prefixPath) # 拷贝一份当前前缀路径列表，避免干扰到下一次循环
            backup_prefixPath.add(item) # 将当前项加入路径中
            temp = copy.deepcopy(backup_prefixPath) # 拷贝一份，便于对频繁项排序
            temp = list(temp)
            temp.sort()
            freqItem = ','.join(temp) # 当前项加入路径中后，便生成了新的频繁项集
            freqItems[freqItem] = headerTable[item][0] # 该频繁项集的支持度取决于最后添加进去那一项在当前FP树中的支持度
                                                       # 即set(['a','b'])的支持度取决于support('b')
            condPattBases = self.findPrefixPath(headerTable[item][1]) # 返回当前项的所有条件模式基
            myCondTree,myHead = self.createFPTree(condPattBases,minSupport) # 基于当前项的所有条件基生成条件FP树
            # 判断生成的条件FP树是否为空，若为空，代表以当前项为起点的所有频繁项都生成完了；否则，表示还可以继续生成
            if myHead != None:
                self._generateFreqItems(myCondTree,myHead,minSupport,backup_prefixPath,freqItems)

    def generateFrequentItem(self,filePath):
        '''生成频繁项集，及其支持度'''
        data = self._loadData(filePath) # 加载原始数据集
        transactionDict = self.createInitSet(data) # 包裹一下原始数据集
        # 生成FP树
        myFPTree,myHeaderTab = self.createFPTree(transactionDict,self.minSupport) # FP树的根节点，头指针表
        FreqItems = {} # 存放频繁项集
        prefixPath = set([]) # 前缀路径，初始为空
        self._generateFreqItems(myFPTree,myHeaderTab,self.minSupport,prefixPath,FreqItems)
        return FreqItems

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
        itemSupport = self.generateFrequentItem(filePath) # 生成频繁项集和其支持度
        print(len(itemSupport))
        ruleSupport, ruleConfidence, ruleLift = self.generateRule(itemSupport) # 生成规则的支持度，置信度，提升度
        return ruleSupport, ruleConfidence, ruleLift,itemSupport


if __name__ == '__main__':
    fp = FPGrowth(minSupport=0.03,minConfidence=0.5)
    ruleSupport, ruleConfidence, ruleLift ,itemSupport = fp.fit(filePath='../data/transaction.xlsx')
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
    with pd.ExcelWriter('../data/result.xlsx') as writer:
        df1.to_excel(writer,encoding='utf-8',index=False,sheet_name='规则表')
        df2.to_excel(writer,encoding='utf-8',index=False,sheet_name='推荐项目及权重')