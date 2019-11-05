# -*- coding: utf-8 -*-
# @Time    : 2019/10/29 12:30
# @Author  : Weiyang
# @File    : CRF_keras.py

#=============================================================================================================
# 本部分代码学习于 https://kexue.fm/archives/5542/comment-page-1#comments ，post存储于./post内

# 基于梯度下降法实现线性链条件随机场：
# 1. 转移矩阵参数：转移特征函数
# 2. 观察值生成标签的参数：状态特征函数
# 3. 损失函数为 负的对数似然函数，目标是最小化损失函数；极大似然估计的角度来理解就是 最大化正的对数似然函数；
#    min{ - log P(Y|X) } = max{ logP(Y|X)}
#=============================================================================================================

from keras.layers import Layer
import keras.backend as K

class CRF_keras(Layer):
    """
    纯Keras实现CRF层
    CRF层本质上是一个带训练参数的loss计算层，因此CRF层只用来训练模型，而预测则需要另外建立模型
    """
    def __init__(self,ignore_last_label=False,**kwargs):
        """ignore_last_label: 定义要不要忽略最后一个标签，起到mask的效果"""
        self.ignore_last_label = 1 if ignore_last_label else 0
        super(CRF_keras,self).__init__(**kwargs)

    '创建转移特征矩阵：转移特征函数及其权重'
    def build(self,input_shape):
        '''input_shape = [batch_size,sequence_length]'''
        self.num_labels = input_shape[-1] - self.ignore_last_label # 计算需要的隐状态个数
        self.trans = self.add_weight(name='crf_trans',
                                     shape=(self.num_labels,self.num_labels),
                                     initializer='glorot_uniform',
                                     trainable=True) # 状态转移特征函数的矩阵，shape=[k,k] k为标签或隐状态的类别数

    '递归计算归一化因子Z(x)'
    def log_norm_step(self,inputs,states):
        """
        1. 递归计算：
        2. 用logsumexp避免溢出
        技巧：通过expand_dims来对齐张量
        """
        states = K.expand_dims(states[0],2) # (batch_size,output_dim,1)
        trans = K.expand_dims(self.trans,0) # (1,output_dim,output_dim)
        output = K.logsumexp(states + trans,1) # (batch_size,output_dim)
        return output + inputs,[output + inputs]

    def path_score(self,inputs,labels):
        """计算目标路径的非规范化概率
        要点：逐标签得分，加上转移概率得分
        技巧：用“预测”点乘“目标”的方法抽取出目标路径的得分
        """
        point_score = K.sum(K.sum(inputs*labels,2),1,keepdims=True) # 逐标签得分
        labels1 = K.expand_dims(labels[:,:-1],3)
        labels2 = K.expand_dims(labels[:,1:],2)
        labels = labels1 * labels2 # 两个错位labels，负责从转移矩阵中抽取目标转移得分
        trans = K.expand_dims(K.expand_dims(self.trans,0),0)
        trans_score = K.sum(K.sum(trans*labels,[2,3]),1,keepdims=True)
        return point_score + trans_score # 两部分得分之和

    def call(self,inputs):
        """CRF本身不改变输出，它只是一个loss"""
        return inputs

    def loss(self,y_true,y_pred):
        """计算对数似然损失函数：-logP(y_1,y_2,...,y_n|x)
        目标y_pred需要one_hot形式"""
        mask = 1 - y_true[:,1:,-1] if self.ignore_last_label else None
        y_true,y_pred = y_true[:,:,:self.num_labels],y_pred[:,:,:self.num_labels]
        init_states = [y_pred[:,0]] # 初始状态
        log_norm,_,_ = K.rnn(self.log_norm_step,y_pred[:,1:],init_states,mask=mask) # 计算Z向量(对数)
        log_norm = K.logsumexp(log_norm,1,keepdims=True) # 计算Z(对数)
        path_score = self.path_score(y_pred,y_true) # 计算分子(对数)
        return log_norm - path_score # 即log(分子/分母)

    def accuracy(self,y_true,y_pred):
        """训练过程中显示逐帧准确率的函数，排除了mask的影响"""
        mask = 1 - y_true[:,:,-1] if self.ignore_last_label else None
        y_true,y_pred = y_true[:,:,:self.num_labels],y_pred[:,:,:self.num_labels]
        isequal = K.equal(K.argmax(y_true,2),K.argmax(y_pred,2))
        isequal = K.cast(isequal,'float32')
        if mask == None:
            return K.mean(isequal)
        else:
            return K.sum(isequal*mask) / K.sum(mask)






























