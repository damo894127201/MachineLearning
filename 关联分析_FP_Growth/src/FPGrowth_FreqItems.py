# -*- coding: utf-8 -*-
# @Time    : 2019/11/14 22:17
# @Author  : Weiyang
# @File    : FPGrowth_FreqItems.py

#======================================================================================================================
# FP-Growth算法：用于高效地发现频繁项集，但不能用于发现关联规则，基于分治算法实现

# 应用案例：
#         你用过搜索引擎吗？输入一个单词或者单词的一部分，搜索引擎就会自动补全查询词项。用户甚至事先都不知道搜索引擎推荐的东西
#         是否存在，反而会去查找推荐词项。我也有过这样的经历，当我输入以“为什么”开始的查询时，有时会出现一些十分滑稽的推荐
#         结果。为了给出这些推荐查询词项，搜索引擎公司的研究人员使用了本模块的FP-Growth算法。他们通过查看互联网上的用词来找出
#         经常在一起出现的词对。这需要一种高效发现频繁集的方法。

# FP-Growth基于Apriori构建，但在完成相同任务时采用了一些不同的技术。这里的任务是将数据集存储在一个特定的称作FP树的数据结构
# 之后发现频繁项集或者频繁项对，即常在一块出现的元素项的集合FP树。这种做法使得算法的执行速度要快于Apriori，通常性能要好两个
# 数量级以上。

# FP-Growth 挖掘频繁项集的速度要比Apriori快，它只需要对数据库进行两次扫描，而Apriori算法对于每个潜在的频繁项集都会扫描数据集
# 判定给定模式是否频繁。在小规模数据集上，这不是什么问题，但当处理更大数据集时，就会产生较大问题。FP-Growth只会扫描数据库两次
# 它发现频繁项集的基本过程如下：
# 1. 构建FP树(Frequent Pattern Tree)，目的是将数据库中的数据压缩到该树上，但保留项集之间的关联信息
#    1. 树节点(Tree Node)：存储该节点的项名；该结点处在该位置上的次数(即不同事务中，该项出现在该位置的次数(排过序)，即支持度，因为
#                          从根节点到该结点其实代表了一个频繁模式，而该结点的次数就是该频繁模式的支持度，显然从根节点开始，
#                          路径中各个结点的支持度是非增的，即递减或不减的)；该结点的父结点(便于从叶节点或中间节点回溯至根节点
#                          ，从而获取前缀路径，构成每个项的条件基)；该结点的子节点(用一个字典来表示，因为可能是多个分支)；
#                          该结点存储项的下一个结点(即下一个存储值为同一个项的，树中的某个结点)
#    2. 根节点(root node)：该结点的项为NULL，即不存储任何项，它的子节点指向，所有经过以各项支持度排序过的事务的第一个元素项；
#    3. FP-Tree：将事务数据表中各个事务数据项按照支持度排序后，把每个事务的数据项按降序(频率)依次插入到一棵以NULL为根节点的数中
#       同时在每个结点出记录该结点出现的支持度；注意只按支持度排序的话，是无法处理支持度相同的项之间的顺序问题，因此需要按照
#       项名的字母表对支持度相同的项再次进行排序，这样可以确保当遇到相同的事务(项序列)时，排序后的项顺序不变；
#    4. 头指针表：一般用字典来存储，字典的key是事务中的每个项(剔除支持度小于指定阈值的项，这一点利用了Apriori算法原理)，
#                 value是一个列表，包含两部分：该项的频率或支持度，该项在树中的
#                 某个节点，该节点中有个指针(链接link)指向FP树中另一个该项的结点，同理另一个该项的结点指向下一个结点，直至
#                 将该项在树中所有的结点都链接起来，最后一个结点的指针为None
# 2. 从FP树中挖掘频繁项集
#    1. 条件模式基(conditional pattern base)：这是对于每个元素项而言的，即条件模式基是以所查找元素项为结尾的FP树路径集合，
#                                             每一条路径其实都是一条前缀路径(prefix path)。
#    2. 前缀路径(prefix path)：介于所查找元素项(位于FP树中，可能是叶节点也可能是非叶节点)与树根节点之间的所有项构成的序列
#    3. 条件FP树(conditional Frequent Pattern Tree)：基于条件模式基构建的FP树，用于抽取频繁项集
#    4. 生成频繁项的过程：
#        构建一个列表来存储前缀路径，如果是递归调用，则将上一步的前缀路径传进去
#       1. 遍历头指针表中的每个项
#       2.   拷贝一份当前前缀路径，将当前项加入备份的前缀路径中；之所以要拷贝一份，是因为如果不拷贝的话，当前循环会把当前项的
#            前缀路径的所有项都添加进去，但这些项对后续循环的其它项而言是多余的；
#       3.   将备份的前缀路径作为新的频繁项集加入到频繁项集列表中 (这一步是生成频繁项集)
#       4.   对每个项，生成其条件基集合
#       5.   对条件基集合(相当于事务集合)，生成条件FP树
#       6.   判断条件FP树是否为空，若为空则停止；
#       7.   否则，递归调用上述步骤1,2,3,4,5，注意传递进去的前缀路径是拷贝的前缀路径
#    5. 频繁项集支持度的计算
#       每个新生成的频繁项集的支持度取决于该频繁项集最后被添加进那个项的支持度，而且这个支持度不是存储全体数据的FP树上的支持度
#       而是基于该项在前缀路径中前一项的条件模式基而生成的条件FP树中的支持度。这是由于在全体数据的FP树中，某项的条件模式基中，
#       该项的支持度是最小的，当基于该条件模式基生成频繁项时，其实是将该项的支持度进行分解了，分解到一个个条件FP树中；


# 算法输入：
# 1. 最小支持度和最小置信度
# 2. 商品交易事务的数据集：每一行表示一条交易事务，和该交易对应的所有商品序列，商品间以 , 隔开，即
#    销售单编号            商品编码序列
#    1317060100203          2BPJ006,4DCS002
#    1317060600067          2BYF034,2BKB006,2BPB031
#    注：在商品编码序列中的商品不重复
# 3. 算法输出
#    频繁项(Items)和其支持度(support) ，用字典来存储
#======================================================================================================================

from FPTreeNode import FPTreeNode
import pandas as pd
from collections import defaultdict
import copy
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')

class FPGrowth(object):
    '''FPGrowth算法挖掘频繁项集'''
    def __init__(self,minSupport):
        self.minSupport = minSupport # 最小支持度

    def _loadData(self,filePath):
        '''加载数据'''
        data = pd.read_excel(filePath,index=False)
        return data

    def createInitSet(self,data):
        '''对数据集处理,包裹为字典类型，并记录每条事务出现支持度'''
        transactionDict = defaultdict(float) # 存储数据集，key是事务，value是事务的支持度
        col1,col2 = data.columns
        numTransaction = len(data.index) # 数据总量
        for id in data.index:
            transaction = data.loc[id][col2].split(',')
            transaction.sort() # 排序一下，便于将相同的事务归为一处
            transactionDict[frozenset(transaction)] += 1 / float(numTransaction)
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
                # 保留支持度大于等于指定阈值的项，这些项便是1-项频繁集
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
                #orderedItems = [v[0] for v in sorted(sorted(localID.items(), key=lambda p:p[0]),key=lambda p:p[1],reverse=True)]
                orderedItems = [v[0] for v in sorted(localID.items(), key=lambda p:(p[0],p[1]), reverse=True)]
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

    def generateFreqItems(self,FPTree,headerTable,minSupport,prefixPath,freqItems):
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
            backup_prefixPath.append(item) # 将当前项加入路径中
            temp = copy.deepcopy(backup_prefixPath)  # 拷贝一份，便于对频繁项排序，使得生成的频繁项有序
            temp.sort()
            freqItem = ','.join(temp) # 当前项加入路径中后，便生成了新的频繁项集
            freqItems[freqItem] = headerTable[item][0] # 该频繁项集的支持度取决于最后添加进去那一项在当前FP树中的支持度
                                                       # 即set(['a','b'])的支持度取决于support('b')
            condPattBases = self.findPrefixPath(headerTable[item][1]) # 返回当前项的所有条件模式基
            myCondTree,myHead = self.createFPTree(condPattBases,minSupport) # 基于当前项的所有条件基生成条件FP树
            # 判断生成的条件FP树是否为空，若为空，代表以当前项为起点的所有频繁项都生成完了；否则，表示还可以继续生成
            if myHead != None:
                self.generateFreqItems(myCondTree,myHead,minSupport,backup_prefixPath,freqItems)

    def fit(self,filePath):
        '''生成频繁项集，及其支持度'''
        logger = logging.getLogger('Fit')
        data = self._loadData(filePath) # 加载原始数据集
        transactionDict = self.createInitSet(data) # 包裹一下原始数据集
        # 生成FP树
        myFPTree,myHeaderTab = self.createFPTree(transactionDict,self.minSupport) # FP树的根节点，头指针表
        logger.info('存放全体数据的FP树的头指针表中含有{}个项'.format(len(myHeaderTab)))
        logger.info('生成的FP树如下：')
        myFPTree.disp(ind=1) # 输出全体数据的FP树
        FreqItems = {} # 存放频繁项集
        prefixPath = [] # 前缀路径，初始为空
        self.generateFreqItems(myFPTree,myHeaderTab,self.minSupport,prefixPath,FreqItems)
        return FreqItems

if __name__ == '__main__':
    fp = FPGrowth(minSupport=0.03)
    freItems = fp.fit(filePath='../data/transaction.xlsx')
    print(len(freItems))