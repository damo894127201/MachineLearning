# -*- coding: utf-8 -*-
# @Time    : 2019/9/24 19:55
# @Author  : Weiyang
# @File    : SVM.py

#=======================================================================================================================
# 支持向量机模型(SVM,Support Vector Machines)
# 1. 概念：支持向量机SVM是一种二类分类模型，是定义在特征空间上的间隔最大化的线性分类器，它是求解能够正确划分训练数据集并且几何
#          间隔最大化的分离超平面。
# 2. 与感知机的区别：感知机只是找到一个分离超平面即可，而SVM确是要找到间隔最大化的那个超平面；感知机是线性分类器，SVM可用于非线性分类
# 3. 支持向量机的学习策略是间隔最大化(几何间隔)，可形式化一个求解凸二次规划问题；
# 4. 支持向量机也等价于 正则化的合页损失函数的最小化问题；

# 模型分类：
# 1. 线性可分支持向量机：硬间隔最大化(几何间隔)
#    给定**线性可分**训练数据集，通过间隔最大化或等价地求解相应的凸二次规划问题，学习得到的分离超平面
#                                 W * X + b = 0
#    以及相应的分类决策函数
#                                 f(X) = sign(W * X + b)
#    称为线性可分支持向量机。

# 2. 线性支持向量机：软间隔最大化，松弛变量，惩罚参数C
#    对于给定的**线性不可分**的训练数据集，通过求解凸二次规划问题，即软间隔最大化问题，得到的分离超平面
#                                 W * X + b = 0
#    以及相应的分类决策函数
#                                 f(X) = sign(W * X + b)
#    称为线性支持向量机。

# 3. 非线性支持向量机：核函数，核技巧
#    从非线性分类训练集，通过核函数与软间隔最大化，或凸二次规划问题，学习得到的分类决策函数
#                                 f(X) = sign(∑α_i * y_i K(X,X_i) + b)
#    称为非线性支持向量机，K(X,Z)是正定核函数。

# 函数间隔 和 几何间隔
# 1. 函数间隔：对于给定的训练数据集T和超平面(w,b)，定义超平面(w,b)(即w*x+b=0,x是多维的)关于样本点(x_i,y_i)的函数间隔为
#                                  γ' = y_i * (w * x_i + b)
#              而超平面关于训练集T的函数间隔为 超平面关于T中所有样本点的函数间隔之最小值。

# 2. 几何间隔：对于给定的训练数据集T和超平面(w,b)，定义超平面(w,b)(即w*x+b=0,x是多维的)关于样本点(x_i,y_i)的几何间隔为
#                                  γ = y_i * (w /||w||* x_i + b/||w||)
#              而超平面关于训练集T的几何间隔为 超平面关于T中所有样本点的几何间隔之最小值。几何间隔不会随着超平面参数成比例
#              的变化而变化，函数间隔会变化。

# 3. 函数间隔和几何间隔 的联系
#    函数间隔可以表示分类预测的正确性及确信度，但是选择间隔最大的超平面，只有函数间隔还不够，因为只要成比例地改变 w 和 b ,例如
#    2w 和 2b，超平面并没有改变，但函数间隔却是原来的2倍。因此，对分离超平面的法向量w施加一些约束，例如规范化，||w||=1，使得
#    间隔是确定的，这时的函数间隔就是几何间隔。

# 线性可分支持向量机公式推导：
# 1. 目标：最大化几何间隔的分离超平面
#                                   max{w,b} γ
#                                   s.t.     y_i * (w /||w||* x_i + b/||w||) >= γ
#    约束条件表示的是超平面(w,b)关于每个训练样本点的几何间隔至少是 γ 。

# 2. 用函数间隔来表示几何间隔：     γ = γ'/||w|| ,上述问题改写为
#                                   max{w,b} γ'/||w||
#                                   s.t.     y_i * (w /||w||* x_i + b/||w||) >= γ'/||w||
#    即：
#                                   max{w,b} γ'/||w||
#                                   s.t.     y_i * (w * x_i + b) >= γ'
#    之所以要用 函数间隔来表示几何间隔，是因为几何间隔是不变的，在优化问题时，我们事先不可能知道这个确定的值，而函数间隔就不同了
#    它的值可以随着超平面参数成比例变化，因此我们便可以固定函数间隔为 1 来 解这个规划问题，固定 函数间隔 γ' = 1，则
#                                   max{w,b} 1/||w||
#                                   s.t.     y_i * (w * x_i + b) >= 1
#    我们可以将上述最大化问题转为下面最小化问题：
#                                   min{w,b} 1/2 * ||w||^2
#                                   s.t.     y_i * (w * x_i + b) >= 1
#    这边得到了我们要求解的凸二次规划问题。

# 3. 凸优化问题：
#    指约束最优化问题：
#                                   min{w} f(w)
#                                   s.t.   g_i(w) <= 0  i = 1,2,...,k
#                                          h_i(w) = 0   i = 1,2,...,m
#    其中，目标函数 f(w) 和 约束函数 g_i(w) 都是R^n 上的连续可微的凸函数，约束函数 h_i(w) 是R^n上的仿射函数。
#    仿射函数：它满足 f(x) = a * x + b ,a∈R^n，b∈R，x∈R^n

#    凸二次规划问题：当目标函数 f(w) 是 二次函数且约束函数g_i(w)是仿射函数时，上述凸优化问题便是凸二次规划问题。

# 4. 运用拉格朗日对偶性：为了求解线性可分支持向量机的最优化问题，将其作为原始最优化问题，应用拉格朗日对偶性，通过求解对偶问题
#                      得到原始问题的最优解，这便是线性可分支持向量机的对偶算法。
#    优点：1. 对偶问题往往更容易求解；2. 自然引入核函数，进而推广到非线性分类问题；

# 5. 构建拉格朗日函数：
#    1. 原始问题的形式：
#                                   min{w,b} 1/2 * ||w||^2
#                                   s.t.     y_i * (w * x_i + b) >= 1
#       将约束条件变为凸优化问题的形式：
#                                   min{w,b} 1/2 * ||w||^2
#                                   s.t.     1 - y_i * (w * x_i + b) <= 0
#    2. 对每一个约束引入拉格朗日乘子α_i >= 0 ,i = 1,2,...,N(N为训练实例的数量)
#    3. 拉格朗日函数如下：
#                                   L(w,b,α) = 1/2 * ||w||^2 + ∑α_i * (1 - y_i * (w * x_i + b))
#                                             = 1/2 * ||w||^2  - ∑α_i * y_i * (w * x_i + b) + ∑α_i
#       其中，α = (α_1,α_2,...,α_N)^T 为拉格朗日乘子向量，我们引入拉格朗日函数的目标是求 :
#                                   max{α}   L(w,b,α)
#       即求α对 L(w,b,α)的极大
#    4. 原始问题的形式为：
#                                   min{w,b}max{α}  L(w,b,α)
#       这是，极小极大问题。
#    5. 原始问题的对偶形式为：
#                                   max{α}min{w,b}  L(w,b,α)
#       这是，极大极小问题。
#    6. 求解流程：
#       1. 求 min{w,b}  L(w,b,α)
#          将 L(w,b,α) 分别对 w,b求偏导并令其等于0，得：
#                                   ▽w L(w,b,α) = w - ∑α_i * y_i * x_i = 0
#                                   ▽b L(w,b,α) = - ∑α_i * y_i = 0
#                                                                             i = 1,2,...,N(N为训练实例的数量)
#          得到：
#                                    w = ∑α_i * y_i * x_i
#                                    ∑α_i * y_i = 0
#          将其代入拉格朗日函数L(w,b,α)中，得：
#                                    L(w,b,α) = -1/2 * ∑i∑j α_i * α_j * y_i * y_j(x_i·x_j) + ∑α_i
#          即                        min{w,b}L(w,b,α) = -1/2 * ∑i∑j α_i * α_j * y_i * y_j(x_i·x_j) + ∑α_i
#                                                                             i,j = 1,2,...,N(N为训练实例的数量)
#       2. 求 min{w,b}L(w,b,α) 对α的极大，即：
#                                    max{α} min{w,b}L(w,b,α)
#          上述便是原始问题的对偶问题，具体形式为：
#                                    max{α} -1/2 * ∑i∑j α_i * α_j * y_i * y_j(x_i·x_j) + ∑α_i
#                                    s.t. ∑α_i * y_i = 0
#                                         α_i >= 0    i = 1,2,...,N(N为训练实例的数量)
#          我们继续将上述结果由求极大转为求极小，就得到与其等价的对偶最优化问题(不是原始问题，而是原始问题对偶问题的对偶问题)
#                                    min{α} 1/2 * ∑i∑j α_i * α_j * y_i * y_j(x_i·x_j) - ∑α_i
#                                    s.t. ∑α_i * y_i = 0
#                                         α_i >= 0    i = 1,2,...,N(N为训练实例的数量)
#          我们的目标是找到一组α = (α_1,α_2,...,α_N)^T 的拉格朗日乘子向量，由其进而求解参数 w 和 b
#      3. 求解分离超平面的参数 w 和 b
#         根据拉格朗日函数的KKT条件，基于拉格朗日函数
#                                    L(w,b,α) = 1/2 * ||w||^2  - ∑α_i * y_i * (w * x_i + b) + ∑α_i
#         以及约束条件
#                                    s.t.     1 - y_i * (w * x_i + b) <= 0
#         则 α = (α_1,α_2,...,α_N)^T 是上述原始问题和对偶问题的解的充分必要条件，即KKT条件是
#                                   ▽w L(w,b,α) = w - ∑α_i * y_i * x_i = 0
#                                   ▽b L(w,b,α) = - ∑α_i * y_i = 0
#                                   α_i * (y_i * (w * x_i + b) - 1) = 0  # KKT的对偶互补条件!!!!!!!!!!!!!!!!!!!!!!!
#                                                                          由此可知，α>0的样点都是支持向量，α=0的点不是
#                                   y_i * (w * x_i + b) - 1 >= 0
#                                   α_i >= 0    i = 1,2,...,N(N为训练实例的数量)
#         对于拉格朗日乘子向量α，存在下标 j，使得α_j > 0,则由
#                                   α_j(y_j * (w * x_j + b) - 1) = 0  i = 1,2,...,N(N为训练实例的数量)
#         可知
#                                    y_j * (w * x_j + b) - 1 = 0
#         由于y_j = ±1 ,则有
#                                    b = y_j - ∑α_i * y_i * (x_i · x_j)
#         此时该点一定为支持向量，因为 y_j * (w * x_j + b) - 1 = 0
#   7. 此时分离超平面可写为
#                                    ∑α_i * y_i * (x · x_i) + b = 0  ，x_i ，y_i是训练样本的实例特征向量和label
#   8. 分类决策函数可写为
#                                    f(x) = sign(∑α_i * y_i * (x · x_i) + b)
#   9. 总结：分类决策函数只依赖于输入 X 和训练样本的输入x_i的内积。


# 线性支持向量机的公式推导
# 1. 线性不可分的含义：训练集中有某些特异点(x_i,y_i) 不能满足函数间隔γ'>=1 的约束条件
# 2. 松弛变量ζ：为解决线性不可分，即某些特异点的函数间隔不满足>=1的条件，为每个样本点引入一个松弛变量ζ_i>=0，使得
#                函数间隔+ζ_i>=1，即函数间隔γ'>=1-ζ_i，也就是
#                                      y_i * (w·x_i + b) >= 1-ζ_i
# 3. 优化的目标函数：我们将每个变量的松弛变量ζ_i，视为误分类错误的支付代价，因此目标函数变为
#                                   min{w,b} 1/2 * ||w||^2 + C*∑ζ_i
#                                   s.t.     y_i * (w * x_i + b) >= 1-ζ_i
#                                            ζ_i >= 0
#    C > 0是惩罚系数
# 4. 线性不可分的线性支持向量机的学习问题变成如下凸二次规划问题，即原始问题变为：
#                                   min{w,b} 1/2 * ||w||^2 + C*∑ζ_i
#                                   s.t.     y_i * (w * x_i + b) >= 1 -ζ_i
#    将约束条件变为凸优化问题的形式：
#                                   min{w,b} 1/2 * ||w||^2 + C*∑ζ_i
#                                   s.t.     1 -ζ_i - y_i * (w * x_i + b) <= 0
#                                            ζ_i >= 0
#    推导形式与线性可分支持向量机相同，只是还需用拉格朗日函数对松弛变量ζ_i求偏导，并置为0，得到如下对偶问题
#                                    min{α} 1/2 * ∑i∑j α_i * α_j * y_i * y_j(x_i·x_j) - ∑α_i
#                                    s.t.    ∑α_i * y_i = 0
#                                            0 <= α_i <= C    i = 1,2,...,N(N为训练实例的数量)
#          我们的目标是找到一组α = (α_1,α_2,...,α_N)^T 的拉格朗日乘子向量，由其进而求解参数 w 和 b

# 非线性支持向量机的对偶问题为：
#                                    min{α} 1/2 * ∑i∑j α_i * α_j * y_i * y_j * K(x_i,x_j) - ∑α_i
#                                    s.t.    ∑α_i * y_i = 0
#                                            0 <= α_i <= C    i = 1,2,...,N(N为训练实例的数量)
#          我们的目标是找到一组α = (α_1,α_2,...,α_N)^T 的拉格朗日乘子向量，由其进而求解参数 w 和 b

# 硬间隔 和 软间隔
# 1. 硬间隔：每个样本点都满足函数间隔γ'>=1
# 2. 软间隔：为每个样本点引入松弛变量ζ_i，使得 函数间隔+ζ_i>=1

# 支持向量、间隔边界 和 分离超平面
# 1. 支持向量：使约束条件 y_i * (w * x_i + b) = 1 成立的点；因为我们定义函数间隔为1，而函数间隔是训练集T中所有距离分离超平面
#              函数间隔最小的点，因此支持向量便是距离分离超平面最近的点。
#              H1：当类别为+1时，w * x_i + b = 1 ，支持向量在此超平面上
#              H2：当类别为-1时，w * x_i + b = -1 ，支持向量在此超平面上
# 2. 间隔：超平面H1与超平面H2之间的距离称为间隔，间隔依赖于分离超平面的法向量w,等于2/||w||
# 3. H1 和 H2 称为间隔边界
# 4. w * x + b = 0 为分离超平面

# 核函数(Kernel Function)：表示将输入从输入空间映射到特征空间得到的特征向量之间的内积。
# 核技巧(Kernel Trick)：通过使用核函数可以学习非线性支持向量机，等价于隐式地在高维的特征空间中学习线性支持向量机。

# 支持向量机的学习算法(学习参数)：SMO算法求解凸二次规划问题
# SMO(Sequential minimal optimization,序列最小最优化算法)要解决如下凸二次规划问题
#                                    min{α} 1/2 * ∑i∑j α_i * α_j * y_i * y_j * K(x_i,x_j) - ∑α_i
#                                    s.t.    ∑α_i * y_i = 0
#                                            0 <= α_i <= C    i = 1,2,...,N(N为训练实例的数量)
#          我们的目标是找到一组α = (α_1,α_2,...,α_N)^T 的拉格朗日乘子向量，由其进而求解参数 w 和 b
# 在上述问题中，变量是拉格朗日乘子，一个变量α_i 对应于一个样本点(x_i,y_i)；变量的总数等于训练样本容量N

# SMO算法与坐标上升算法(Coordinate Ascent)的相似处
# 坐标上升算法每次通过更新多元函数中的一维，经过多次迭代直到收敛来达到优化函数的目的。简单地讲，
# 就是不断地选中一个变量做一维最优化直到函数达到局部最优点；

# SMO算法思路：如果所有变量的解都满足此最优化问题的KKT条件，那么这个最优化问题的解就得到了。因为KKT条件是该最优化问题的充要条件。
# 1.选择两个变量α1 和 α2 ，固定其它变量(α3,...,αn)；
# 2. 针对这两个变量构建一个二次规划问题(目标函数是关于α1和α2的二次函数)：即在已知α1和α2的条件下，求使得目标函数最小的α1和α2；
#    这个二次规划问题关于这两个变量的解应该更接近原始二次规划问题的解，因为这会使得原始二次规划问题的目标函数值变得更小；该问题
#    是原始问题的子问题，是可以通过解析方法求解(二次函数求最值，取偏导数为0的极值点函数值)
# 3. SMO算法将原始优化问题不断分解为一个个固定两个变量的子问题，并对子问题求解析解，进而达到求解原问题的目的。
# 4. 该子问题有两个变量：
#    1. 一个是违反KKT条件最严重的变量: 先遍历由支持向量训练实例构成的集合(即0<α< C)，如果都满足KKT条件; 再遍历其余训练实例
#                                     找到一个不满足KKT条件的实例；如果都满足KKT，则退出循环，解已求出；
#    2. 一个是由约束条件自动确定的：先遍历支持向量训练实例构成的集合，再遍历其余训练实例，
#                                  如果还不达要求，则更换第一个变量，继续遍历一遍；
# 5. 将两变量的最优化问题变为单变量的最优化问题；
# 6. 子问题的两个变量中，只有一个自由变量，即违反KKT条件的那个变量α1，当α1确定时，α2也随之确定，可以用α1来表示α2，这样就把
#     二元的目标函数视为一元的函数，然后求极值，就得到了 更新 前后变量之间的关系:
#                                   α2{new,unc} = α2{old} + y_2(E1-E2)/η
#     其中，η = K11 + K22 - 2K12 = ||φ(x1) - φ(x2)||^2 ，K是核函数，φ(x)是输入空间到特征空间的映射；
#     Ei,i=1,2 ,是预测值y_pred与真实值y_i的差值 ； 子问题同时更新两个变量；

# SMO算法分为两部分：
# 1. 求解两个变量的二次规划问题
# 2. 选择变量的启发式方法

# SMO算法具体流程：
# 1. 设置最大迭代轮次，开启迭代，作为最外层的循环
# 2. 第二层循环分为两个并列的两层子循环：
#    1. 第二层循环：从支持向量集中选择第一个拉格朗日乘子变量；第三层循环：进而选择第二个拉格朗日乘子变量
#    2. 第二层循环：从非支持向量集中选择第一个拉格朗日乘子向量；第三层循环：进而选择第二个拉格朗日乘子向量
# 3. 判断全体拉格朗日乘子向量前后差异是否达到指定阈值，即将前后的拉格朗日乘子向量，对应的分量相减，再取平方求和开根号
#=======================================================================================================================

import numpy as np
from kernels import *
from tqdm import tqdm

class SVM(object):
    '''支持向量机模型'''
    def __init__(self,X,Y,C=1,kernel=None,gap=1e-3,max_iter=200):
        self.C = C # 惩罚参数，C越大，模型越容易过拟合；C越小，模型容易欠拟合；
                   # C值大时，对误分类的惩罚增大，因为C直接与分类损失相乘，C值大时，会放大分类损失，导致模型会过度学习；
                   # C值小时，对误分类的惩罚减小；最优化目标有两个：1. 使1/2||w||^2尽量小，即间隔尽量大；2. 使误分类点尽可能少
                   # C是调和二者的系数；
        self.gap = gap # 用来判断是否收敛的阈值,即前后两次更新所有alpha之间差值的几何平均值是否小于这个gap
        self.max_iter = max_iter # 迭代次数的最大值

        # 选择核函数
        if kernel is None:
            self.kernel = LinearKernel() # 默认是线性核
        elif kernel == "poly":
            self.kernel = PolyKernel() # 多项式核
        elif kernel == "RBF":
            self.kernel = RBF() # 高斯核

        self.X = X  # 训练数据的特征,X = np.array([[value,...],...])
        self.Y = np.array(Y) # label ,Y = [label,...]
        self.n_samples = X.shape[0] # 训练样例数
        self.n_features = X.shape[1] # 样例的特征数
        # 用来存储 训练实例两两之间的 核函数值 ，即两两之间在高维特征空间的内积
        self.K = np.zeros((self.n_samples,self.n_samples))

        for i in range(self.n_samples):
            self.K[:,i] = self.kernel(self.X,self.X[i,:]) # 计算当前实例与其余所有实例的核函数值

        self.alpha = np.zeros(self.n_samples) # 拉格朗日乘子初始化，全部置为0
        self.b = 0 # 超平面的偏置
        self.Error = [-label for label in self.Y] # 用来保存每个样例点的Error=g(x_i) - y_i ，加速计算
                                                  # 由于初始时，所有样例的α都是0，因此初始时预测误差正好是 负的label
        self.support_vectors = [] # 用来保存训练集中属于支持向量的实例的索引
        self.non_support_vectors = [] # 用来保存训练集中 非支持向量的实例的索引

    def _calculate_g(self,x_index:int):
        '''
        根据拉格朗日乘子alpha先求得W,继而计算 y_i = W * x_i + b的值，即g(x_i)
        分类决策函数 f(x) = sign(W * x_i + b) = sign(g(x_i))
        这里，只计算括号内的值g(x_i)
        '''
        return np.sum(self.alpha * self.Y * self.K[:,x_index]) + self.b

    def _selectFirstAlpha(self,alpha_index:int) -> bool:
        '''选择第一个变量alpha1，本函数用于判断选定的第一个α是否满足条件'''
        # α= 0
        if self.alpha[alpha_index] == 0:
            if self.Y[alpha_index] * self._calculate_g(alpha_index) >= 1:
                return True
        # 0 <α< C
        elif 0 < self.alpha[alpha_index] < self.C:
            if self.Y[alpha_index] * self._calculate_g(alpha_index) == 1:
                return True
        # α= C
        elif self.alpha[alpha_index] == self.C :
            if self.Y[alpha_index] * self._calculate_g(alpha_index) <= 1:
                return True
        return False # 当前α不满足KKT条件

    def _calculate_Error(self,index):
        '''计算g(x) - y_index，即预测值与真实输出之差，是用于选中第二个变量α的辅助函数'''
        return self._calculate_g(index) - self.Y[index]

    def _selectSecondAlpha(self,first_alpha_index:int) -> int:
        '''
        第二个变量alpha2的选择，具体的：
        根据第一个拉格朗日乘子alpha1 和 随机选择的第二个拉格朗日乘子alpha2，
        判断alpha2是满足要求；
        返回 alpha2的索引，作为更新的第二个变量alpha
        '''
        # 目标是找到使得alpha2变换最大的那个
        # 由于alpha2 依赖于|E1 -E2|，选择使|E1 -E2|最大的那个alpha2
        # 由于alpha1已定，E1也确定了
        E1 = self.Error[first_alpha_index] # 先前已经计算过
        # 搜索使|E1 -E2|最大的那个alpha2
        index = float("inf")  # 存储使|E1 -E2|最大的那个alpha2的索引
        max_abs = 0 # 存储最大的|E1 -E2|
        for alpha2_index in range(self.n_samples):
            # 第一个拉格朗日乘子的索引不能与第二个拉格朗日乘子重合
            if alpha2_index != first_alpha_index:
                E2 = self._calculate_Error(alpha2_index)
                if abs(E1 - E2) > max_abs :
                    max_abs = abs(E1 - E2)
                    index = alpha2_index
        return index

    def _getLH(self,alpha1,alpha2,y1,y2):
        '''
        基于alpha1_old和alpha2_old，确定第一个变量alpha1新值的范围：

        根据 alpha1_old 和alpha2_old 的值，和 y1 和 y2的值，
        获取 alpha1_new 的最小值L和最大值H;
        '''
        # y1 = y2
        if y1 == y2:
            L1 = max(0,alpha1 + alpha2 - self.C)
            H1 = min(self.C,alpha1 + alpha2)
            return L1,H1
        else:
            L1 = max(0,alpha1 - alpha2)
            H1 = min(self.C,self.C + alpha1 - alpha2)
            return L1,H1

    def _clip(self,alpha,L,H):
        '''剪辑alpha的值 到 L 和 H之间'''
        if alpha < L:
            return L
        elif alpha > H:
            return H
        return alpha

    def _calculate_alpha1(self,alpha1_index,alpha2_index,eta):
        '''更新第一个拉格朗日乘子变量alpha1，返回更新前后的值'''
        E1 = self._calculate_Error(alpha1_index)
        E2 = self._calculate_Error(alpha2_index)
        alpha1_old = self.alpha[alpha1_index]
        alpha1_new = alpha1_old + self.Y[alpha1_index] * (E2 -E1) / eta
        return alpha1_new,alpha1_old

    def _calculate_alpha2(self,alpha1_index,alpha1_old,alpha1_new,alpha2_index):
        '''更新第二个拉格朗日乘子变量alpha2，返回更新前后的值'''
        alpha2_old = self.alpha[alpha2_index]
        alpha2_new = alpha2_old + self.Y[alpha1_index] * self.Y[alpha2_index] * (alpha1_old - alpha1_new)
        return alpha2_new,alpha2_old

    def _calculate_b(self,alpha1_index,alpha1_old,alpha1_new,alpha2_index,alpha2_old,alpha2_new):
        '''更新偏置b'''
        b1_new = - self.Error[alpha1_index] - self.Y[alpha1_index] * self.K[alpha1_index,alpha1_index] * (alpha1_new - alpha1_old) - \
                 self.Y[alpha2_index] * self.K[alpha2_index,alpha1_index] * (alpha2_new - alpha2_old) + self.b
        b2_new = - self.Error[alpha2_index] - self.Y[alpha1_index] * self.K[alpha1_index,alpha2_index] * (alpha1_new - alpha1_old) - \
                 self.Y[alpha2_index] * self.K[alpha2_index,alpha2_index] * (alpha2_new - alpha2_old) + self.b
        if 0 < alpha1_new < self.C:
            self.b = b1_new
        elif 0 < alpha2_new < self.C:
            self.b = b2_new
        else:
            self.b = 0.5 * (b1_new + b2_new)

    def _update_Error(self,alpha1_index,alpha2_index):
        '''更新 alpha1_index 和 alpha2_index对应的误差self.Error'''
        # 更新E1 和 E2
        E1,E2 = - self.Y[alpha1_index],- self.Y[alpha2_index]
        for index in self.support_vectors:
            E1 += self.Y[index] * self.alpha[index] * self.K[alpha1_index,index]
            E2 += self.Y[index] * self.alpha[index] * self.K[alpha2_index, index]
        E1 += self.b
        E2 += self.b

        # 更新
        self.Error[alpha1_index] = E1
        self.Error[alpha2_index] = E2

    def smo(self):
        '''SMO算法求解支持向量机的凸二次规划问题'''
        # 在最大迭代次数内求解
        for now_iter in tqdm(range(self.max_iter)):
            if now_iter == 0:
                # 第一轮选取第一个变量alpha时，支持向量的集合为空
                # 因此将训练集中所有数据的索引都存入非支持向量的集合中
                self.non_support_vectors = list(range(self.n_samples))

            # 拷贝一份未更新的alpha拉格朗日乘子向量
            alpha_prior = np.copy(self.alpha)

            # 开始选择第一个拉格朗日乘子变量alpha1
            # 先从支持向量的集合中选择
            if len(self.support_vectors) != 0:
                # 这里没有比较各个alpha，哪一个违反KKT条件最严重，只是遍历
                for alpha1_index in self.support_vectors:
                    # 判断当前alpha是否违反KKT条件
                    # 不违反KKT条件，则选择下一个
                    if not self._selectFirstAlpha(alpha1_index):
                        continue

                    # 基于第一个变量，选择第二个拉格朗日乘子变量
                    alpha2_index = self._selectSecondAlpha(alpha1_index)

                    # 判断η是否大于0
                    eta = self.K[alpha1_index,alpha1_index] + self.K[alpha2_index,alpha2_index]\
                          - 2.0 * self.K[alpha1_index,alpha2_index]
                    if eta <= 0:
                        continue
                    # 更新alpha1
                    alpha1_new, alpha1_old = self._calculate_alpha1(alpha1_index,alpha2_index,eta)
                    # 获取alpha1_new的范围
                    L,H = self._getLH(alpha1_old,self.alpha[alpha2_index],self.Y[alpha1_index],self.Y[alpha2_index])
                    # 对alpha1进行剪辑
                    alpha1_new = self._clip(alpha1_new,L,H)
                    # 更新alpha1
                    self.alpha[alpha1_index] = alpha1_new

                    #更新alpha2
                    alpha2_new,alpha2_old = self._calculate_alpha2(alpha1_index,alpha1_old,alpha1_new,alpha2_index)
                    self.alpha[alpha2_index] = alpha2_new

                    # 更新支持向量集 self.support_vectors
                    # 如果alpha1_new = 0，说明alpha1_index对应的实例不再是支持向量，要删除；
                    # alpha1_index本身就是从支持向量集中选出的
                    if alpha1_new == 0:
                        # 从支持向量集中删除
                        self.support_vectors.remove(alpha1_index)
                        # 将其加入非支持向量集，注意别丢失数据
                        self.non_support_vectors.append(alpha1_index)
                    # alpha2_index是否是从支持向量集中选出的，我们需要判断
                    if alpha2_new > 0 :
                        # 如果alpha2_index不在支持向量集中
                        if alpha2_index not in self.support_vectors:
                            # 将其加入支持向量集
                            self.support_vectors.append(alpha2_index)
                            # 将其从非支持向量集中删除
                            self.non_support_vectors.remove(alpha2_index)
                    # 更新偏置b
                    self._calculate_b(alpha1_index, alpha1_old, alpha1_new, alpha2_index, alpha2_old, alpha2_new)
                    # 更新 误差self.Error
                    self._update_Error(alpha1_index, alpha2_index)
            # 如果支持向量的集合为空，则从非支持向量的集合中选择
            # 这里没有比较各个alpha，哪一个违反KKT条件最严重，只是遍历
            # 遍历非支持向量集
            for alpha1_index in self.non_support_vectors:
                # 判断当前alpha是否违反KKT条件
                # 不违反KKT条件，则选择下一个
                if self._selectFirstAlpha(alpha1_index):
                    continue

                # 基于第一个变量，选择第二个拉格朗日乘子变量
                alpha2_index = self._selectSecondAlpha(alpha1_index)

                # 判断η是否大于0
                eta = self.K[alpha1_index,alpha1_index] + self.K[alpha2_index,alpha2_index]\
                      - 2.0 * self.K[alpha1_index,alpha2_index]
                if eta <= 0:
                    continue
                # 更新alpha1
                alpha1_new, alpha1_old = self._calculate_alpha1(alpha1_index,alpha2_index,eta)
                # 获取alpha1_new的范围
                L,H = self._getLH(alpha1_old,self.alpha[alpha2_index],self.Y[alpha1_index],self.Y[alpha2_index])
                # 对alpha1进行剪辑
                alpha1_new = self._clip(alpha1_new,L,H)
                # 更新alpha1
                self.alpha[alpha1_index] = alpha1_new

                #更新alpha2
                alpha2_new,alpha2_old = self._calculate_alpha2(alpha1_index,alpha1_old,alpha1_new,alpha2_index)
                self.alpha[alpha2_index] = alpha2_new

                # 更新支持向量集 self.support_vectors
                # 如果alpha1_new = 0，说明alpha1_index对应的实例不再是支持向量，要删除；
                # alpha1_index本身就是从非支持向量集中选出的
                if alpha1_new > 0:
                    # 从支持向量集中删除
                    self.non_support_vectors.remove(alpha1_index)
                    # 将其加入支持向量集，注意别丢失数据
                    self.support_vectors.append(alpha1_index)
                # alpha2_index是否是从支持向量集中选出的，我们需要判断
                if alpha2_new > 0 :
                    # 如果alpha2_index不在支持向量集中
                    if alpha2_index not in self.support_vectors:
                        # 将其加入支持向量集
                        self.support_vectors.append(alpha2_index)
                        # 将其从非支持向量集中删除
                        self.non_support_vectors.remove(alpha2_index)
                # 更新偏置b
                self._calculate_b(alpha1_index, alpha1_old, alpha1_new, alpha2_index, alpha2_old, alpha2_new)
                # 更新 误差self.Error
                self._update_Error(alpha1_index, alpha2_index)

            # 判断是否收敛，即前后两轮迭代之间 所有对应alpha之间的差值的平方和是否小于指定的阈值self.gap
            gap = np.linalg.norm(self.alpha - alpha_prior)
            # 如果达到指定阈值，且迭代次数达到指定次数，则停止迭代，跳出
            if gap < self.gap and now_iter > int(0.5 * self.max_iter):
                break

    def fit(self):
        '''训练SVM'''
        self.smo()

    def _calculate_new_g(self,x):
        '''
        根据拉格朗日乘子alpha先求得W,继而计算 y_i = W * x_i + b的值，即g(x_i)
        分类决策函数 f(x) = sign(W * x_i + b) = sign(g(x_i))
        这里，只计算括号内的值g(x_i)
        x 是一条预测数据
        '''
        Kernel_value = self.kernel(self.X,x) # 计算当前数据x 与训练数据的核函数值矩阵
        return np.sum(self.alpha * self.Y * Kernel_value) + self.b

    def predict(self,X):
        '''预测，X = np.array([[value,value,...],...])'''
        n_samples = X.shape[0]
        result = np.zeros(n_samples)
        print()
        print("拉格朗日乘子向量是: ")
        print(self.alpha)
        # 预测
        for i in range(n_samples):
             result[i] = np.sign(self._calculate_new_g(X[i])) # 正类预测为1，负类为-1
        return result