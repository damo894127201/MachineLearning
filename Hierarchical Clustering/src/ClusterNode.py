# -*- coding: utf-8 -*-
# @Time    : 2019/11/5 19:56
# @Author  : Weiyang
# @File    : ClusterNode.py

#==================================================================================================================
# 聚类结点类型：用于存储每条数据，便于构造聚类树
#==================================================================================================================

import numpy as np

class ClusterNode(object):
    '''聚类结点类型：所有的真实数据都存储在叶节点上，所有的聚类中心都存储在非叶节点上'''

    def __init__(self,value,ids=[],id=None,left=None,right=None,distance=-1,count=-1,check=False):
        self.value = value  if type(value) == np.ndarray else np.array(value)
                            # 该结点存储的数据，合并结点时等于原来结点值的算术平均值，即各个维度取平均
                            # 同时也指代聚类中心的位置 ，类型为np.array
        self.ids = ids  # 当前结点包含的id序列，即以该结点为聚类中心的类别所包含的数据id
        self.id = id # 当前结点的id
        self.left = left  # 合并得到该结点的左子节点
        self.right = right # 合并得到该结点的右子节点
        self.distance = distance # 两个子结点的距离
        self.count = count # 该结点所包含的叶结点的个数，即样本的个数
        self.check = check # 标识符，用于遍历时指示该结点是否被遍历过,即该结点是否已经被归并到其它类中
                           # True表示已被遍历过，False表示未被遍历过

    def toString(self):
        '''输出节点内容'''
        print("data:",self.value,' ','node id :',self.id,' ','node ids :',self.ids,'left.ids:',
              self.left.ids if self.left != None else None,' ','right.ids: ',self.right.ids if self.right != None else None,
              ' ','left and right distance:',self.distance,' ','data count: ',self.count)