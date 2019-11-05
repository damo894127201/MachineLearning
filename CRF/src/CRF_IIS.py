# -*- coding: utf-8 -*-
# @Time    : 2019/11/1 21:40
# @Author  : Weiyang
# @File    : CRF_IIS.py

#==============================================================================================================
# IIS算法实现CRF
#==============================================================================================================

class CRF_IIS(object):
    '''条件随机场：IIS算法实现'''

    def __init__(self,num_latent_states,**kwargs):
        self.num_latent_states = num_latent_states # 隐状态类别数
        # 转移特征矩阵 state_trans_matrix，是一个 num_latent_states 阶 的矩阵，其中每个元素为 :
        #                         M_i(y_i-1,y_i|x) = exp{ W_i(y_i-1,y_i|x) }
        #                         W_i(y_i-1,y_i|x) = ∑_{k=1...K} w_k * f_k(y_i-1,y_i)
        if 'state_trans_matrix' not in kwargs:
            # 如果没有提供状态转移矩阵，则随机初始化
        self.state_trans_matrix =





































