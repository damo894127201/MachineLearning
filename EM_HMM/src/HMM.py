# -*- coding: utf-8 -*-
# @Time    : 2019/10/9 14:07
# @Author  : Weiyang
# @File    : HMM.py

# ================================================================================================================
# 隐马尔可夫模型(Hidden Markov Model,HMM)：生成模型
# 概念：隐马尔可夫模型描述由隐藏的马尔可夫链随机生成观测序列的过程。具体而言，隐马尔可夫模型是关于时序的概率模型，描述
#       由一个隐藏的马尔可夫链 随机生成 不可观测的状态随机序列，再由各个状态生成一个观测 从而产生 观测随机序列 的过程。
#       隐藏的马尔可夫链随机生成的状态的序列，称为状态序列(state sequence)；每个状态生成一个观测，而由此产生的 观测的
#       随机序列，称为 观测序列(observation sequence)。序列的每一个位置又可以看作是一个时刻。

# 马尔可夫过程：是满足 无后效性 的随机过程。假设一个随机过程中，t_n 时刻的状态 x_n 的条件分布，仅仅与其前一个状态 x_n-1
#               有关，即P(x_n | x1,x2,...,x_n) = P(x_n | x_n-1)，则称其为 马尔可夫过程。
# 马尔可夫链：时间和状态的取值都是离散的马尔可夫过程，称为 马尔可夫链。

# 在隐马尔可夫模型中，隐状态x_i 对于观测者而言是不可见的，观测者能观测到的只有每个隐状态x_i对应的输出 y_i ，而观测状态y_i
# 的概率分布仅仅取决于对应的隐状态x_i。

# 隐马尔可夫模型的三要素
# 1. 状态转移概率矩阵A
# 2. 观测概率矩阵B
# 3. 初始状态概率向量π
# π和 A决定状态序列，确定了隐藏的马尔可夫链，生成不可观测的状态序列；B确定了如何从状态生成观测，与状态序列综合确定了如何产生
# 观测序列，因此隐马尔可夫模型λ可以用三元符号表示，即λ= (A,B,π)

# 隐马尔可夫模型的两个基本假设
# 1. 齐次马尔可夫性假设：
#    假设隐藏的马尔可夫链，在任意时刻t的状态，只依赖于其前一时刻的状态，与其它时刻的状态及观测无关，也与时刻t无关；
# 2. 观测独立性假设
#    假设任意时刻的观测只依赖于该时刻的马尔可夫链的状态，与其它观测及状态无关；
#    MEMM(最大熵隐马尔可夫模型则打破了观测独立性假设，)

# 隐马尔可夫模型的三个基本问题
# 1. 概率计算问题：给定隐马尔可夫模型λ和观测序列O，计算在模型λ下观测序列O出现的概率P(O|λ)
#    1. 前向算法：前向概率
#    2. 后向算法：后向概率是一个条件概率，因为从后往前
# 2. 学习问题：已知观测序列O，估计模型λ的参数A,B,π，使得在该模型下观测序列概率P(O|λ)最大，即用极大似然估计的方法估计参数。
#    1. 监督学习：
#                训练数据包括：观测序列和对应的状态序列，模型参数可通过统计频率估算出来
#    2. 无监督学习：
#                训练数据，只有观测序列，模型参数通过 鲍姆-韦尔奇(Baum-Welch)算法计算出来，该算法由EM算法实现。
# 3. 预测问题：也称解码问题(Decoding)，已知模型λ和观测序列O，求对给定观测序列O，最大的状态序列I，即P(I|O,λ)
#             预测算法可通过 近似算法 和 维特比算法(Viterbi algorithm)实现

# 隐马尔可夫模型的用途：序列标注(词性标注，分词，分词也可以视为序列标注)
# 隐马尔可夫模型可以用于标注，这时隐状态对应着标记。标注问题是给定观测的序列，预测其对应的标记序列。

# 序列标注任务：在给定 不定长观测序列的基础上，尝试找到决定该观测序列的另一组序列。
# ================================================================================================================

# 本部分代码学习于 https://zhuanlan.zhihu.com/p/75406198 ，笔者改动的部分主要集中在 矩阵计算
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')


class HMM(object):
    '''隐马尔可夫模型'''

    def __init__(self, num_latent_states, num_observation_states, **kwargs):
        self.num_latent_states = num_latent_states  # 隐状态个数
        self.num_observation_states = num_observation_states  # 观测状态个数
        # 初始状态概率分布π init_prob_dist
        if 'init_prob_dist' not in kwargs:
            self.init_prob_dist = np.random.random((self.num_latent_states,))  # 如果没有提供初始概率分布π，则随机初始化
            self.init_prob_dist = self.init_prob_dist / np.sum(self.init_prob_dist)  # 归一化概率，确保初始概率和为1
        else:
            self.init_prob_dist = kwargs['init_prob_dist']

        # 状态转移矩阵A state_trans_matrix
        if 'state_trans_matrix' not in kwargs:
            # 如果没有提供状态转移矩阵A，则初始化为各个隐状态间相互转移的概率相同
            self.state_trans_matrix = np.ones((self.num_latent_states, self.num_latent_states))
            self.state_trans_matrix = self.state_trans_matrix / self.num_latent_states
        else:
            self.state_trans_matrix = kwargs['state_trans_matrix']

        # 观测概率分布B，或发射矩阵 emission_matrix
        if 'emission_matrix' not in kwargs:
            # 如果没有提供观测概率分布B，则随机初始化，但确保同一个隐状态下的各个发射概率和为1
            self.emission_matrix = np.random.random((self.num_latent_states, self.num_observation_states))
            sum_row = np.sum(self.emission_matrix, axis=1)  # 各行概率之和
            self.emission_matrix = self.emission_matrix / sum_row[:, None]  # 在sum_row中扩充一个维度，确保按行除
        else:
            self.emission_matrix = kwargs['emission_matrix']

    '''概率计算：前向算法'''
    def forward(self, input):
        """
        input = [o1,o2,...,ot] 观测状态序列，注意这里的观测值都是用整数顺序编码的
        一、目标：给定参数λ= (init_prob_dist,state_trans_matrix,emission_matrix)，用前向算法求p(o|λ)
                 定义 在给定隐马尔可夫模型λ，到时刻t的从前往后的部分观测序列为 o1,o2,...,ot，
                 且时刻t的隐状态为q_i，的概率为前向概率，
                 记作：α_t(i) = P(o1,o2,...,ot,i_t=q_i | λ)
        二、假设：
            1. 齐次Markov假设：p(i_t+1 | o1,o2,...,ot,i_1,i_2,...,i_t) = p(i_t+1 | i_t)
                              任意时刻的隐状态只与前一时刻的隐状态有关，与其它时刻的状态以及观测无关
            2. 观测独立性假设：p(o_t+1 | o1,o2,...,ot,i_1,i_2,...,i_t,i_t+1) = p(o_t+1 | i_t+1)
                              任意时刻的观测状态只与该时刻的马尔可夫链的隐状态有关，与其它时刻的观测及状态无关
        三、求解
            1. 标记前向概率：
               alpha_t(i) = p(o1,...,o_t,i_t=q_i)
            2. 初始化：
               alpha_1(i) = p(o1,i_1=q_i)
                          = p(i_1=q_i) * p(o1 | i_1=q_i)
                          = init_prob_dist[q_i] * emission_matrix[q_i,o1]
               alpha_1 = init_prob_dist[:] * emission_matrix[:,o1] ，表示 时刻1 且 观测为o1 时，状态为各个隐状态的前向概率
               注意：这里是position_wise乘法，不是点乘(向量内积，矩阵乘法)
               shape = (num_latent_states,)*(num_latent_state,) = (num_latent_state,)
            3. 迭代关系：
               alpha_t(i)   = p(o1,...,ot,i_t = q_i)
               alpha_t+1(j) = p(o1,...,ot,o_t+1,i_t+1 = q_j)
                            = p(o1,...,ot,i_t+1 = q_j) * p(o_t+1 | i_t+1 = q_j)
                            = sum_i{p(o1,...,ot,i_t = q_i,i_t+1 = q_j)} * p(o_t+1 | i_t+1 = q_j) # 观测独立性假设
                            = sum_i{p(o1,...,o_t,i_t = q_i) * p(i_t+1 = q_j| i_t = q_i )} * p(o_t+1 | i_t+1 = q_j)
                                                                                                 # 齐次马尔可夫假设
                            = sum_i {alpha_t[i] * state_trans_matrix[q_i,q_j]} * emission_matrix[q_j,o_t+1]
               alpha_t+1 = np.dot(alpha_t[None,:],state_trans_matrix)[0] * emission_matrix[:,o_t+1]
               注意，这里是position_wise乘法(*)，不是点乘(.*)
               shape = ((1,num_latent_state) .* (num_latent_state,num_latent_state)) * (num_latent_state,)
                     = (1,num_latent_state)[0] * (num_latent_state,) # [0] 目的在于降维
                     = (num_latent_state,) * (num_latent_state,) # 对应位置相乘
                     = (num_latent_state,)
            4. 求P(O|λ)
               alpha_T = p(O,i_T = q_i)
               P(O|λ) = sum_i { p(O,i_T = q_i)}
                       = sum_i(alpha_T)
        """
        # 初始化
        T = len(input)  # 观测序列长度
        alpha = []  # 存储各个时刻的前向概率集合，每个时刻由于隐状态不一样，又有多个前向概率
        # 注意，是从前往后存储，取数据时要注意
        alpha_t = self.init_prob_dist * self.emission_matrix[:, input[0]]  # 时刻0的前向概率集合，
        # 在下文则指代上一时刻的前向概率集合
        alpha.append(alpha_t)
        # 迭代计算各个时刻的前向概率,从第二个时刻开始，下标为1
        for t in range(1, T):
            alpha_t = np.dot(alpha_t[None, :], self.state_trans_matrix)[0] * self.emission_matrix[:, input[t]]
            alpha.append(alpha_t)
        # 计算，在给定模型λ下，观测序列inputs的概率
        p_o = np.sum(alpha_t, axis=0)
        # 返回p_o,alpha用于 baum-welch算法
        return p_o, np.array(alpha)

    '''概率计算：后向算法'''
    def backward(self, input):
        """
        input = [o1,o2,...,ot] 观测状态序列，注意这里的观测值都是用整数顺序编码的
        一、目标：给定参数λ= (init_prob_dist,state_trans_matrix,emission_matrix)，用后向算法求p(o|λ)
                 定义 给定隐马尔可夫模型λ，求在时刻t，状态为q_i的条件下，从t+1到T时刻的从后往前的部分观测序列为
                 ot+1,ot+2,...,oT，的概率为后向概率，
                 记作 β_t(i) = P(ot+1,ot+2,...,oT | i_t=q_i ,λ)，后向概率是一个条件概率，因为是从后往前
        二、假设：
            1. 齐次Markov假设：p(i_t+1 | o1,o2,...,ot,i_1,i_2,...,i_t) = p(i_t+1 | i_t)
                              任意时刻的隐状态只与前一时刻的隐状态有关，与其它时刻的状态以及观测无关
            2. 观测独立性假设：p(o_t+1 | o1,o2,...,ot,i_1,i_2,...,i_t,i_t+1) = p(o_t+1 | i_t+1)
                              任意时刻的观测状态只与该时刻的马尔可夫链的隐状态有关，与其它时刻的观测及状态无关
        三、求解
            1. 标记
               这里设序列长度为T
               beta_t(i) = p(ot+1,ot+2,...,oT | i_t = q_i,λ)
            2. 初始化
               beta_T-1(i) = p(oT | i_T-1 = q_i) # 表示倒数第二个时刻的后向概率
                           = sum_j {p(oT ,i_T = q_j | i_T-1 = q_i,λ)}
                           = sum_j {p(oT | i_T = q_j,i_T-1 = q_i,λ) * p(i_T = q_j | i_T-1 = q_i,λ)} # 齐次马尔可夫假设
                           = sum_j {p(oT | i_T = q_j) * p(i_T = q_j | i_T-1 = q_i,λ)} # 观测独立性假设
                           = sum_j {state_trans_matrix[q_i,q_j] * emission_matrix[q_j,oT]}
               beta_T-1 = np.dot(state_trans_matrix,emission_matrix[:,oT])
               shape = (num_latent_state,num_latent_state) .* (num_latent_state,)
                     = (num_latent_state,)
               注：beta_T = [1,...,1] shape = (num_latent_state,) 。因为时刻T已是观测序列末尾，后面无观测数据，因此全部设为1
            3. 迭代关系
               beta_t+1(j) = p(ot+2,...,o_T | i_t+1 = q_j)
               beta_t(i)   = p(ot+1,....,o_T | i_t = q_i)
                           = sum_j { p(ot+1,...,o_T, i_t+1 = q_j | i_t = q_i)}
                           = sum_j { p(ot+2,...,o_T,i_t+1 = q_j | i_t = q_i) *
                                    p(ot+1 | ot+2,...,o_T , i_t+1 = q_j , i_t = q_i)} # 联合概率拆分
                           = sum_j { p(i_t+1 = q_j | i_t = q_i) * p(o_t+2,..,o_T | i_t = q_i,i_t+1 = q_j)
                                     * p(ot+1 | i_t+1 = q_j)}                         # 齐次马尔可夫假设和观测独立性假设
                           = sum_j { p(i_t+1 = q_j | i_t = q_i) * p(o_t+2,..,o_T |i_t+1 = q_j) * p(ot+1 | i_t+1 = q_j)}
                                          概率图阻断原理：p(o_t+2,...,o_T|i_t=q_i,i_t+1=q_j) = p(o_t+2,...,o_T|i_t+1=q_j)
                           = sum_j { state_trans_matrix[q_i,q_j] * beta_t+1(j) * emission_matrix[q_j,o_t+1]}
                           = sum_j { state_trans_matrix[q_i,q_j] * emission_matrix[q_j,o_t+1] * beta_t+1(j)}
               beta_t = np.dot(state_trans_matrix * emission_matrix[:,o_t+1],beta_t+1[None,:].T)[:,0]
                                                                                                [:,0]目的在于降维
               注意：这里np.dot()里的第一项是逐项相乘，是position_wise相乘(*) ，不是点乘(.*)，点乘是矩阵乘法
               shape = np.dot((num_latent_states,num_latent_states) * (num_latent_states,) , (num_latent_states,1))[:,o]
                     = np.dot((num_latent_states,num_latent_states),(num_latent_states,1))[:,0]
                     = (num_latent_states,1)[:,0]
                     = (num_latent_states,)
            4. 求P(O|λ)
                beta_1  = p(o2,...,oT|i_1 = q_i)
                P(O|λ) = sum_i { p(O,i_1=q_i)}
                        = sum_i { p(O|i_1=q_i) * p(i_1 = q_i)}
                        = sum_i { p(o1,o2,...,oT | i_1 = q_i) * p(i_1=q_i)}
                        = sum_i { p(o1 | o2,...,oT,i_1 = q_i) * p(o2,...,oT | i_1 = q_i) * p(i_1=q_i) }
                        = sum_i { p(o1 | i_1 = q_i) * p(o2,...,oT | i_1 = q_i) * p(i_1=q_i)} # 观测独立性假设
                        = sum_i {emission_matrix[q_i,o1] * beta_1(i) * init_prob_dist[q_i]}
                        = np.dot(beta_1,emission_matrix[:,o1] * init_prob_dist)
                        = np.dot(beta_1[None,:],emission_matrix[:,o1] * init_prob_dist)
                shape = np.dot((num_latent_states,),(num_latent_states,) * (num_latent_states,))
                      = np.dot((num_latent_states,),(num_latent_states,)) # 这一步其实是将数组做矩阵乘法
                      = (1,)                                              # 即维度是(num_latent_states,)的数组视为
                                                                          # (1，num_latent_states)
        """
        # 初始化
        T = len(input)  # 观测序列长度
        beta = []  # 存储各个时刻的 后向概率集合，每个时刻的后向概率，因隐状态取值不同，同样也是一个集合
        # 注意，顺序是从后往前存储，本模块取数据时要注意，下文，返回时需要翻转
        beta_t = np.array([1.] * self.num_latent_states)  # T时刻的所有后向概率都为1，该值在下文指代下一时刻的后向概率集合
        beta.append(beta_t)
        # 迭代计算各个时刻的后向概率集合,从后往前
        for t in range(0, T - 1)[::-1]:
            beta_t = np.dot(self.state_trans_matrix * self.emission_matrix[:, input[t + 1]], beta_t[None, :].T)[:, 0]
            beta.append(beta_t)
        # 计算，在给定模型λ下，观测序列inputs的概率
        p_o = np.dot(beta[-1][None, :], self.emission_matrix[:, input[0]] * self.init_prob_dist)
        # 返回的p_o，beta用于baum-welch算法
        return p_o, np.array(beta[::-1])  # 翻转

    '''学习算法：Baum-Welch算法'''
    def baum_welch(self, inputs, conv_loss=1e-8):
        """
        inputs = [[o1,o2,...,],...] 观测序列的集合,注意这里的观测值都是用整数顺序编码的
        一、目标：利用Baum-Welch算法无监督训练HMM
        二、迭代公式(EM算法)：
            模型用lambda = (A,B,π)表示 ，I表示隐状态序列，O表示观测序列，目标是求使得Q函数最大的lambda
            lambda_t+1 = argmax_lambda sum_I{ log(p(O,I|lambda)) * p(I | O,lambda_t)}
                       = argmax_lambda sum_I{ log(p(I,O|lambda)) * p(I,O|lambda_t)/p(O|lambda_t)}
                       = argmax_lambda sum_I{ log(p(I,O|lambda)) * p(I,O|lambda_t)}
            lambda_t+1 = (init_prob_dist_t+1,state_trans_matrix_t+1,emission_matrix_t+1)
            设Q函数为：
                Q(lambda,lambda_t) = sum_I{ log(p(I,O|lambda)) * p(I,O|lambda_t)}
            设序列长度为T
                p(I,O|lambda) = init_prob_dist[i_1] * emission_matrix[i_1,o1] * state_trans_matrix[i_1,i_2]
                                * emission_matrix[i_2,o2] * state_trans_matrix[i_2,i_3] * emission_matrix[i_3,o3]
                                * state_trans_matrix[i_3,i_4] * .....* state_trans_matrix[i_T-1,i_T]
                                * emission_matrix[i_T,o_T]
            因此Q函数改写为：
                Q(lambda,lambda_t) = sum_I{ p(I,O|lambda_t) * log(init_prob_dist[i_1])}
                                     + sum_I{ p(I,O|lambda_t) * sum_t{ log(state_trans_matrix[i_t,i_t+1])}}
                                     + sum_I{ p(I,O|lambda_t) * sum_t{ log(emission_matrix[i_t,o_t])}}
            这里涉及到约束优化问题：
                np.sum(init_prob_dist[i]) = 1
                np.sum(state_trans_matrix,axis=1) = [1,1,...,1]  shape = (num_latent_states,)
                np.sum(emission_matrix,axis=1) = [1,..,1]  shape = (num_latent_states,)
            求解带约束的最优化问题，需要构造拉格朗日函数，对待求参数求导取0.这里，需要注意的是，上述Q函数中极大化的三项是
            单独出现在Q函数中的，因此，只需分别对这三项构造拉格朗日函数求极大化即可。
            得出的结果便是Baum_Welch算法的参数更新公式。

            1.  定义一些辅助概率
                给定模型λ和观测O，在时刻t处于状态q_i的概率，记为：
                gama_t(i) = p(i_t = q_i|O,λ)
                          = p(i_t = q_i,O | λ) / p(O|λ)
                          = alpha_t(i) * beta_t(i) / sum_j{ alpha_t(j) * beta_t(j)}

                在时刻t处于各种状态的概率：
                gama_t = alpha * beta / np.sum(alpha * beta,axis=1)
                       = alpha * beta / p(O|λ)
                shape  = (num_latent_states,)

                在各个时刻处于各种状态的概率：
                gama = [gama_1,gama_2,...,gama_T]
                shape = (T,num_latent_states)

                给定模型λ和观测O，在时刻t处于状态q_i,且在时刻t+1处于状态q_j的概率，记为：
                X_t[i,j] = p(i_t = q_i,i_t+1 = q_j | O,λ)
                         = p(i_t = q_i,i_t+1 = q_j ,O |λ) / p(O|λ)
                         = p(i_t = q_i,i_t+1 = q_j ,O |λ) / sum_i sum_j{ p(i_t = q_i,i_t+1 = q_j ,O |λ)}
                         = alpha_t(i) * state_trans_matrix[q_i,q_j] * emission_matrix[q_j,o_t+1] * beta_t+1(j)
                           / sum_i sum_j {alpha_t(i) * state_trans_matrix[q_i,q_j] * emission_matrix[q_j,o_t+1] * beta_t+1(j)}

                X_t = (alpha_t[:,None] * state_trans_matrix * emission_matrix[:,o_t+1] * beta_t+1)
                      / np.sum(alpha_t[:,None] * state_trans_matrix * emission_matrix[:,o_t+1] * beta_t+1)
                注: X_t表示，在t时刻和t+1时刻所有可能的取值，即这两个时刻都可以取隐状态的所有取值，
                    因此 X_t的shape= (num_latent_states,num_latent_states)

                各个时刻的当前时刻和后一时刻的各种状态取值，有T-1个时刻，最后一时刻没有
                X = [X_1,X_2,...,X_T-1]
                shape = (T-1,num_latent_states,num_latent_states)

            2.  Baum-Welch算法参数更新公式
                (1) init_prob_dist
                    init_prob_dist[q_i] = gama_1[i]
                    init_prob_dist = gama_1
                (2) state_trans_matrix
                    state_trans_matrix[q_i,q_j] = sum_(t=1～T-1)X_t[q_i,q_j] / sum_(t=1～T-1)gama_t[i]
                                                = np.sum(X[:,q_i,q_j]) / np.sum(gama[:-1,q_i])
                    state_trans_matrix = np.sum(X,axis=0) / np.sum(gama[:-1],axis=0)  这里是巧妙处!
                (3) emission_matrix
                    gama_t[j] = p(i_t = q_j|O,λ)
                    emission_matrix[q_j,o_t=v_k] = sum_(t=1～T) p(i_t=q_j,o1,...o_t-1,o_t=v_k,o_t+1,..o_T|λ)
                                                    / sum_(t=1～T)p(i_t=q_j,O|λ)
                                                 = sum_(t=1～T) {p(i_t=q_j,o_t=v_k|O,λ)p(O',o_t=v_k|λ)} # O'表示除去o_t时刻的观测值v_k
                                                        / sum_(t=1～T){p(i_t=q_j|O,λ)p(O|λ)}
                    显然，下面不好处理了，o_t = v_k 不好通过矩阵处理，用循环来处理
                    # emission_from_gama.shape = (num_latent_states,num_observation_states)
                    emission_from_gama = np.zeros((num_latent_states,num_observation_states))
                    for gama_t ,o_t in zip(gama,O):
                        # gama_t.shape = (num_latent_states,)
                        emission_from_gama[:,o_t] += gama_t
                    emission_matrix = emission_from_gama / np.sum(gama,axis=0)[:,None]
                    np.sum(gama,axis=0).shape = (num_latent_states,)
        """
        logger = logging.getLogger('Baum-Welch')
        epochs = 1  # 记录迭代期

        while True:
            # 创建临时存储发射矩阵、状态转移矩阵和初始概率的变量，用于存储当前迭代期参数的值，用于判断参数是否收敛
            init_prob_dist = np.zeros((self.num_latent_states,))
            state_trans_matrix = np.zeros((self.num_latent_states, self.num_latent_states))
            emission_matrix = np.zeros((self.num_latent_states, self.num_observation_states))

            # 循环遍历观察序列集合的每一个观察序列
            for input_item in inputs:
                p_o_f, alpha = self.forward(input_item)  # 获取当前观察序列的前向概率集合
                p_o_b, beta = self.backward(input_item)  # 获取当前观察序列的后向概率集合
                # 给定模型λ和观测O，在各个时刻处于各种状态的概率的集合
                gama = alpha * beta / p_o_f  # p_o_f与p_o_b相等，都指代P(O|λ)
                # 各个时刻的当前时刻和后一时刻的各种状态取值，有T-1个时刻，最后一时刻没有
                X = []
                # 遍历观察序列的各个时刻
                for t in range(len(input_item) - 1):
                    X_t = (alpha[t][:, None] * self.state_trans_matrix * self.emission_matrix[:, input_item[t + 1]]
                           * beta[t + 1]) / np.sum(
                        alpha[t][:, None] * self.state_trans_matrix * self.emission_matrix[:, input_item[t + 1]] * beta[t + 1])
                    X.append(X_t)
                X = np.array(X)
                # 更新参数
                init_prob_dist += gama[0]  # 累加新的初始概率分布
                state_trans_matrix += np.sum(X, axis=0) / np.sum(gama[:-1], axis=0)[:,None] # 累加新的转移概率矩阵
                # 更新发射概率矩阵
                emission_from_gama = np.zeros((self.num_latent_states, self.num_observation_states))
                for gama_t, o_t in zip(gama, input_item):
                    # gama_t.shape = (num_latent_states,)
                    emission_from_gama[:, o_t] += gama_t
                emission_matrix += emission_from_gama / np.sum(gama, axis=0)[:, None]
            # 由于上面累加了每个观测序列的各个参数的更新值，因此要求取平均，作为新的参数更新值
            init_prob_dist /= len(inputs)
            state_trans_matrix /= len(inputs)
            emission_matrix /= len(inputs)

            # 记录当前参数更新差值，用于判断是否停止迭代
            l2_differences = np.sum(np.power(init_prob_dist - self.init_prob_dist, 2)) + np.sum(
                np.power(state_trans_matrix -
                         self.state_trans_matrix, 2)) + np.sum(np.power(emission_matrix - self.emission_matrix, 2))

            # 更新参数
            self.init_prob_dist = init_prob_dist
            self.state_trans_matrix = state_trans_matrix
            self.emission_matrix = emission_matrix
            #logger.info('epochs:{}\tdifferences{}'.format(epochs, l2_differences))

            # 判断是否收敛,
            if l2_differences <= conv_loss:
                logger.info('Training Finished!')
                break
            epochs += 1

    '''预测算法：维特比算法'''
    def viterbi(self, inputs):
        '''
        inputs = [[o_1,o_2,...,o_T],...],注意这里的观测值都是用整数顺序编码的
        一、目标：利用viterbi算法解码
        二、引入两个变量 delta 和 psi
            1. 定义在时刻t 状态为i的所有路径(i_1,i_2,...,i_t)中概率最大值为：
               delta_t(i) = max_(i_1,i_2,...,i_t-1) P(i_t = i,i_t-1,...,i_1,o_t,...,o_1 | λ)  i = 1,...,N (有N种隐状态)
            2. 定义在时刻t 状态为i的所有路径(i_1,i_2,...,i_t-1，i)中，概率最大的路径的第t-1个节点为：
               psi_t(i) = arg  max_(1<=j<-N) [delta_t-1(j) * state_trans_matrix[j,i]], i = 1,...,N (有N种隐状态)
               用于从最优路径的终端回溯找到其余最优节点
        三、维特比算法：
            1. 初始化：
               时刻1，状态为i的路径概率最大值
               delta_1(i) = init_prob_dist[q_i] * emission_matrix[q_i,o_1]
               时刻1，所有状态的路径概率最大值
               delta_1 = init_prob_dist * emission_matrix[:,o_1]

               psi_1(i) = 0 , 时刻1前面没有节点，因为时刻1是初始节点
            2. 迭代公式
               对于t=2,3,...,T
               delta_t(i) = max_(1<=j<=num_latent_states) [deta_t-1(j) * state_trans_matrix[q_j,q_i]] * emission_matrix[q_i,o_t]

               psi_t(i) = argmax_(1<=j<=num_latent_states) deta_t-1(j) * state_trans_matrix[q_j,q_i]]
               这个式子可以带上最后一项emission_matrix[q_i,o_t]]，argmax和最后一项没关系

            3. 终止：
               P* = max_(1<=i<=num_latent_states) delta_T(i)
               i*_T = argmax_(1<=i<=num_latent_states) delta_T(i)
            4. 最优路径回溯
               对于 t = T-1,T-2,..,1
               i*_t = psi_t+1(i*_t+1)
        '''
        logger = logging.getLogger('Viterbi')
        logger.info('start decoding...')
        routes = []  # 存储每个观测序列对应的隐状态序列
        for input_item in inputs:
            delta, psi, route = [], [], []
            # 初始化
            delta_1 = self.init_prob_dist * self.emission_matrix[:, input_item[0]]  # 时刻1，所有状态的路径概率最大值
            # shape = (num_latent_states,)
            psi_1 = np.zeros((self.num_latent_states,))
            delta.append(delta_1)  # 存储各个时刻,所有状态的路径概率最大值,shape = (T,num_latent_states)
            psi.append(psi_1)  # 存储各个时刻，所有状态的路径概率最大的前一个状态,shape=(T+1,num_latent_states)
            # 迭代
            for t in range(1, len(input_item)):
                # 求当前时刻下，所有状态的路径的概率
                iter_func = delta[-1][:, None] * self.state_trans_matrix * self.emission_matrix[:, input_item[t]]
                # 求当前时刻下，状态为i的路径的概率最大值，由于状态未知，因此是所有状态
                delta_t = np.max(iter_func, axis=0)
                # 求使得当前时刻状态为i的路径的概率最大的前一个时刻的状态,由于状态未知，因此是所有状态
                psi_t = np.argmax(iter_func, axis=0)
                delta.append(delta_t)
                psi.append(psi_t)
            # 最优路径回溯
            route_T = np.argmax(delta[-1])  # 求最后时刻T所有可能的状态中，概率最大的那个状态，即为最后一时刻的状态
            route.append(route_T)
            # 逆向回溯
            for t in range(len(input_item) - 1)[::-1]:
                route_t = psi[t + 1][route[-1]]
                route.append(route_t)
            route = route[::-1]
            routes.append(route)
        logger.info('Decoding finished!')
        return routes


if __name__ == '__main__':

    def myPrint(array,num):
        '''array: 待输出的数组或矩阵 ； num: 表示缩进的制表符个数'''
        for index,item in enumerate(array):
            print('\t' * num, item)

    print()
    print('本部分代码学习于 https://zhuanlan.zhihu.com/p/75406198 ，笔者改动的部分主要集中在 矩阵计算 ')
    print()
    print()

    # 第一部分  概率计算测试，测试实例来自 李航 《统计学习方法》第二版 P200
    print('第一部分：概率计算')
    print()
    print('\t测试实例来自 李航 《统计学习方法》第二版 P200')
    print()
    # 初始化状态概率向量
    init_prob_dist = np.array((0.2, 0.4, 0.4))
    # 状态转移矩阵
    state_trans_matrix = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    # 发射矩阵
    emission_matrix = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])

    # 初始化隐马尔可夫参数
    hmm = HMM(num_latent_states=3, num_observation_states=2, init_prob_dist=init_prob_dist,
              state_trans_matrix=state_trans_matrix, emission_matrix=emission_matrix)
    # 输入状态序列
    inputs = [0, 1, 0]

    # 前向算法
    p_O, alpha = hmm.forward(inputs)
    print('\t前向算法\tp(O):', p_O)
    print('\t前向算法\talpha:')
    myPrint(alpha,5)
    print()

    # 后向算法
    p_O, beta = hmm.backward(inputs)
    print('\t后向算法\tp(O):', p_O)
    print('\t后向算法\tbeta:')
    myPrint(beta,5)
    print()

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # 第二部分  学习算法：Baum-Welch算法
    print('第二部分：学习算法 Baum-Welch算法')
    print()
    # 随机初始化学习
    hmm = HMM(num_latent_states=3, num_observation_states=2)

    # 给定训练序列
    hmm.baum_welch([[0, 1, 0], [1, 1, 0, 1, 0, 0, 0, 1]])

    # 输出三参数
    print('\t初始概率分布:')
    myPrint(hmm.init_prob_dist,5)
    print('\t状态转移矩阵:')
    myPrint(hmm.state_trans_matrix,5)
    print('\t观测概率矩阵:')
    myPrint(hmm.emission_matrix,5)
    print()

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # 第三部分 预测算法: 维特比算法  测试实例来自 李航 《统计学习方法》第二版 P210

    print('第三部分 预测算法: 维特比算法')
    print()
    print('\t测试实例来自 李航 《统计学习方法》第二版 P210')
    print()

    # 初始化状态概率向量
    init_prob_dist = np.array((0.2, 0.4, 0.4))
    # 状态转移矩阵
    state_trans_matrix = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    # 发射矩阵
    emission_matrix = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])

    # 初始化隐马尔可夫参数
    hmm = HMM(num_latent_states=3, num_observation_states=2, init_prob_dist=init_prob_dist,
              state_trans_matrix=state_trans_matrix, emission_matrix=emission_matrix)

    # 指定观测序列
    decode_res = hmm.viterbi([[0, 1, 0]])
    # 输出序列解码结果
    print('\tViterbi decode result:', decode_res)