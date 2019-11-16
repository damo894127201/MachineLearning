# -*- coding: utf-8 -*-
# @Time    : 2019/11/15 13:33
# @Author  : Weiyang
# @File    : FPTreeNode.py

#=====================================================================================================================
# FP 树的结点类型
#=====================================================================================================================

class FPTreeNode(object):
    '''FPTreeNode'''
    def __init__(self,nameValue,numOccur,parentNode):
        self.name = nameValue # 该结点的项名，一般为字符串
        self.support = numOccur # 该结点处于FP树中这个位置的次数或支持度，一般初始化为某个值
                              # 当向FP树中添加新事务时，如果遇到相同的前缀，则增加相应的值，比如{a,b,c,d,e}该前缀出现过5次
                              # 则结点d的self.count初始化时为5，当遇到新事务{a,b,c,d,f,g}时，假设该事务出现的次数为3，则
                              # self.count增加3，调用下面的inc()方法即可实现
        self.nodeLink = None #指向下一个相似结点的指针
        self.parent = parentNode #指向父节点
        self.children = {} #指向所有子节点，分支可能有多个，用字典来存

    def inc(self,numOccur):
        '''当一条事务一条事务扫描时，遇到相似的事务项序列前缀，增加该结点的支持度'''
        self.support += numOccur

    def disp(self,ind=1):
        '''用于输出FP树结构'''
        print('  '*ind,self.name,' ',self.support)
        for child in self.children.values():
            child.disp(ind+1)