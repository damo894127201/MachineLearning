# -*- coding: utf-8 -*-
# @Time    : 2019/9/13 14:05
# @Author  : Weiyang
# @File    : TreeNode.py

##############################################
# 决策树的节点类型
##############################################

class TreeNode:
    '''决策树的节点类型：决策树的内部结点或叶节点'''
    def __init__(self,feature_index = None,threshold = None,value = None,true_branch = None,false_branch = None):
        self.true_branch = true_branch # 左节点，代表左子树，表示当前节点的判断条件为true的子树或叶结点
        self.false_branch = false_branch # 右节点，代表右子树，表示当前节点的判断条件为false的子树或叶结点
        self.value = value # 如果当前节点是叶节点，该值表示节点内的样本类别或样本y的均值
        self.threshold = threshold # 当前节点用于决策的特征的阈值
        self.feature_index = feature_index # 当前节点用于决策的特征的索引