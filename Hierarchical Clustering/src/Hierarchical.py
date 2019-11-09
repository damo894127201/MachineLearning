# -*- coding: utf-8 -*-
# @Time    : 2019/11/5 19:12
# @Author  : Weiyang
# @File    : Hierarchical.py

#======================================================================================================================
# 层次聚类之聚合聚类(Agglomerative): 结果输出一个类别之间的树状图

# 层次聚类：聚合聚类(自下而上) 和 分裂聚类(自上而下)
# 层次聚类假设类别之间存在层次结构，将样本聚到层次化的类中。由于每个样本只属于一个类，因此层次聚类属于硬聚类。

# 聚合聚类：
# 聚合聚类开始将每个样本各自分到一个类；之后将相距最近的两类合并，建立一个新的类，重复此操作直到满足停止条件；得到层次化的类别

# 分裂聚类：
# 分裂聚类开始将所有样本分到一个类；之后将已有类中相距最远的样本分到两个新的类，重复此操作直到满足停止条件；得到层次化的类别。

# 聚合聚类的具体过程：
# 对于给定的样本集合，开始将每个样本分到一个类；然后按照 一定规则，例如类间距离最小，将最满足规则条件的两个类进行合并；如此，
# 反复进行，每次减少一个类，直到满足停止条件，如所有样本聚为一类。

# 聚合聚类需要预先确定下面三个要素：
# 1. 距离或相似度：闵可夫斯基距离、马哈拉诺比斯、相关系数、夹角余弦
# 2. 合并规则：类间距离最小，类间距离可以是最短距离、最长距离、中心距离、平均距离
# 3. 停止条件：类的个数达到阈值(极端情况类的个数是1，即最终所有样本聚为一类)、类的直径超过阈值(类内相距最远的两个样本的距离)

# 层次聚类的优点
# 1. 一次性得到聚类树，后期再分类无需重新计算
# 2. 相似度规则容易定义
# 3. 可以发现类别的层次关系

# 层次聚类的缺点
# 1. 计算复杂度高，不适合数据量大的
# 2. 算法很可能形成链状
#======================================================================================================================

from ClusterNode import ClusterNode
import numpy as np
import copy
import queue
import matplotlib.pyplot as plt

class Hierarchical(object):
    '''层次聚类之聚合聚类: 距离度量采用欧式距离；合并规则为距离最近的两个类(类中心)进行合并；停止条件为 聚为一个类'''

    def distance(self,x,y):
        '''
        计算两个节点的欧式距离
        x , y 是 ClusterNode类型
        '''
        return np.sqrt(np.sum([value*value for value in (x.value - y.value)]))

    def minDist(self,dataset):
        '''计算所有结点中距离最小的节点对'''
        mindist = 1000 # 存储最小距离
        x,y = 0,0 # 存储最小距离的两个结点索引

        for i in range(len(dataset) - 1): # 从前往后计算，要留出最后一个位置的元素给内部循环
            # 略过已经被归并过的节点
            if dataset[i].check == True:
                continue
            # 计算当前结点与后续每个结点的距离
            # 由于是从前往后遍历dataset，两个节点的距离又是对称的，因此只需遍历当前结点后续的结点即可
            for j in range(i+1,len(dataset)):
                # 略过已经被归并过的节点
                if dataset[j].check == True:
                    continue
                dist = self.distance(dataset[i],dataset[j])
                if dist < mindist:
                    mindist = dist
                    x,y = i,j
        # 返回最小距离，以及相应两个结点的索引
        return mindist,x,y

    def fit(self,data):
        '''执行聚类
        data = np.array([[value,value,..],..]),待聚类的数据集
        '''
        # 将输入的数据元素转化为结点，并存入结点的列表
        dataset = [ ClusterNode(value=item,id='node:'+str(i),ids=['node:'+str(i)],count=1) for i,item in enumerate(data)]
        length = len(dataset) # 样本的个数
        Backup = copy.deepcopy(dataset) # 备份数据
        # 开始聚类
        label_count = 1
        while True:
            mindist,x,y = self.minDist(dataset) # 获取当前轮次距离最小的两个类别(类中心)的距离，以及其索引
            dataset[x].check = True # 标识已访问过
            dataset[y].check = True
            newID = copy.deepcopy(dataset[x].ids) # 新类别包含的数据的id列表 ，ids
            newID.extend(dataset[y].ids)
            newCenter = (dataset[x].value+dataset[y].value)/2 # 新类的聚类中心
            newCount = dataset[x].count + dataset[y].count # 新类包含的叶节点个数，即样本个数
            id = 'Label:'+str(label_count) # 新类别的id
            # 将新类别添加到dataset中，参与下轮的聚类
            newClassNode = ClusterNode(value=newCenter,id=id,ids=newID,left=dataset[x],right=dataset[y],distance=mindist,
                                       count=newCount)
            # 将新类加入数据集
            dataset.append(newClassNode)
            # 当新类包含的样本个数等于数据集的样本个数时，表示聚类停止，即所有样本聚为一个类
            if newCount == length:
                break
            label_count += 1

        # 循环输出每个聚类结点的信息
        for node in dataset:
            node.toString()
        # 返回聚类后的结点集合，以及备份的原始的数据集
        return dataset,Backup

    def show(self,dataset,num):
        '''
        输出聚类结果：聚类树
        由于我们希望在二维平面中输出聚类树，因此需要给每个结点赋予相应的横纵坐标，这个坐标与样本数据无关，只与聚类的类别层次有关
        num 表示 聚类树根节点的横坐标
        '''
        plt.figure(1)
        showqueue = queue.Queue() # 创建显示结点信息的队列
        # 将根节点加入队列
        showqueue.put(dataset[len(dataset) - 1]) # dataset中最后加入的结点一定是根节点，即包含所有样本的结点
        showqueue.put(num) # 存入根结点的横坐标
        # 队列不空循环出队
        while not showqueue.empty():
            currentNode = showqueue.get() # 当前绘制的结点
            i = showqueue.get() # 当前绘制结点中心的横坐标
            left = i - (currentNode.count)/2  # 当前结点的左子节点的横坐标
            right = i + (currentNode.count)/2 # 当前结点的右子节点的横坐标

            # 为当前结点的类别中心添加数据标签
            plt.text(i, currentNode.distance + 0.3, currentNode.id)
            # 遍历当前结点的左子节点
            if currentNode.left != None:
                x = [left,right] # 左节点横坐标，右节点横坐标
                y = [currentNode.distance,currentNode.distance] # 用distance来做纵坐标，原因在于距离从根节点往下是逐渐减小的
                                                                # 越靠近根节点的类别，其左右子节点的类别的距离越大
                plt.plot(x,y) # 显示左右子节点 ，即将左右子节点用横线连接起来,画水平线

                x = [left,left] # 左节点横坐标，左节点的左右子节点的横坐标
                y = [currentNode.distance,currentNode.left.distance] # 左节点纵坐标，左节点的左右子节点的纵坐标
                plt.plot(x,y) # 显示当前结点与其左子节点的关系，即将左子节点与当前节点连接起来，画垂直线

                # 往队列中添加当前节点的左节点
                showqueue.put(currentNode.left)
                showqueue.put(left)
            # 遍历当前结点的右子节点
            if currentNode.right != None:
                x = [right,right]
                y = [currentNode.distance,currentNode.right.distance]
                plt.plot(x,y)
                showqueue.put(currentNode.right)
                showqueue.put(right)
        plt.show()

if __name__ == '__main__':
    print("代码参考并学习于: https://blog.csdn.net/weixin_41958939/article/details/83218634")
    # 构建测试数据，采用李航《统计学习方法第二版》P262例题
    data = np.array([[0,7,2,9,3],
                     [7,0,5,4,6],
                     [2,5,0,8,1],
                     [9,4,8,0,5],
                     [3,6,1,5,0]])
    #data = np.random.randint(1, 100, size=100).reshape(20,5)
    model = Hierarchical()
    result,_ = model.fit(data)
    model.show(result,20)