# -*- coding: utf-8 -*-
# @Time    : 2019/9/20 15:53
# @Author  : Weiyang
# @File    : KDNode.py

#====================================================
# kd树的节点
# 主要记录每个节点的信息，包括：
# 1. 存储于当前节点的数据
# 2. 数据对应的标签
# 3. 当前节点的左子节点
# 4. 当前节点的右子节点
# 5. kd树构建过程中，当前节点划分的维度索引(注，维度从1开始计数)
# 6. 当前节点在kd树中深度(注，根节点的深度为0)
#====================================================

class KDNode:
    '''kd树节点'''
    def __init__(self,feature_value,label,left_branch=None,right_branch=None,father=None,split_axis=None,depth=0):
        self.feature_value = feature_value # 存储当前节点的特征值向量，如[value,...]
        self.label = label # 当前节点的数据对应的label 或 y值
        self.left_branch = left_branch # 当前节点的左子节点
        self.right_branch = right_branch # 当前节点的右子节点
        self.father = father # 当前节点的父节点
        self.split_axis = split_axis # 当前节点划分的维度索引
        self.depth = depth # 当前节点的深度