# -*- coding: utf-8 -*-
# @Time    : 2019/9/19 22:15
# @Author  : Weiyang
# @File    : KNN_kdTree.py

#=============================================================================================================
# KNN(K-nearest neighbor K近邻算法)
# KNN假设给定一个训练数据集，其中的实例类别已知。分类时，对新的实例，根据其K个最近邻的训练实例的类别
# 通过多数表决等方式进行预测。

# KNN属于：分类和回归算法，做分类时，返回K近邻中的多数类；做回归时，返回K近邻中各个样本点的均值
# KNN没有显示的学习过程，无需训练。
# KNN实际上利用训练数据集对特征向量空间进行划分。

# KNN三要素：K值的选择，距离度量及分类决策规则
# 1. K值的选择：K值一般取一个比较小的数，根据交叉验证法来选取最优的K值
# 2. 距离的度量：欧式距离，曼哈顿距离，Lp距离
# 3. 分类决策规则：多数表决

# KNN实现方法：线性扫描 和 kd树
# 1. 线性扫描：就是计算输入的新实例与每一个训练实例的距离，从中找出距离最小的K个训练实例，用这个K个
#              实例的多数类，作为新实例的类别；或用这K个实例的均值，作为回归的预测值。
# 2. kd树(kd tree)方法：当训练集很大时，线性扫描计算非常耗时，因此不可取；为提高K近邻的搜索效率，考虑
#                       用特殊的结构存储训练数据，以减少计算距离的次数。实现方法很多，这里用kd树

# kd树：kd树是一种对k维空间中的实例点进行存储以便对其进行快速检索的树形数据结构。kd树是二叉树，表示对
#       k维空间的一个划分。构造kd树相当于不断地用垂直于坐标轴的超平面将k维空间切分，构成一系列的k维
#       超矩形区域。kd树的每个节点对应于一个k维超矩形区域。

# KNN算法实现：
# 1. 构造kd树：
#    1. 选择切分轴的顺序和切分点：从特征向量的低纬度向高维度循环遍历即可，对深度为j的结点，其对应的切分轴L为L=(j mod k) + 1
#                                这里特征向量是k维的，从1开始计数；根节点的深度j=0，因此根节点的切分轴L=(0 mode k) +1 = 1
#                                深度为k-1的节点的切分轴为第k个轴，即特征向量最后一维度；维度值从1开始
#    2. 构造根节点：选择特征向量的第一个维度为切分轴，计算训练数据的第一个维度值的中位数，以该中位数为切分点，将所有第一个
#                   维度值小于中位数的数据划分到左节点，大于中位数的的数据划分到右节点；根节点的取值是第一个维度值是该中位数
#                   的训练实例，并且记下根节点对应的切分轴索引。如果训练数据为偶数个，即中位数为两个数的平均，我们可以随意取
#                   一个数作为中位数，但绝不能取训练数据中不存在的点；
#    3. 递归下去，直到所有数据都存在节点中，终止时的节点便是叶节点；
#    4. 这是个不断切分的过程，实则是对k维空间的划分，每次选取的切分轴，以及过切分点且垂直于切分轴的超平面将子区域分割为两个子
#       区域。这时，剩余的实例被分到两个子区域，直到子区域内没有实例时终止。在此过程中，将实例点保存到相应的节点上。

# 2. 搜索kd树找寻最近点：
#    1. 找出包含目标点x的叶节点：从根节点出发，递归地向下访问kd树。若目标点x在当前维的坐标小于切分点的坐标，则移动到左子节点，
#                               否则移动到右子节点。直到子节点为叶节点为止。
#    2. 以此叶节点为“当前最近点”
#    3. 递归地向上回退，在每个节点进行以下操作：
#               1. 如果该节点保存的实例点比当前最近点距离目标点更近，则以该实例点为“当前最近点”；
#               2. 当前最近点一定存在于该节点的一个子节点对应的区域。检查该子节点的父节点的另一子节点对应的区域是否有更近的点。
#                  具体地，检查另一子节点对应的区域是否与以目标点为球心，以目标点与“当前最近点”间的距离为半径的超球体相交。
#                    1. 如果相交，可能在另一个子节点对应的区域内存在距目标点更近的点，移动到另一个子节点。接着，递归地进行最近邻搜索
#                    2. 如果不相交，向上回退
#    4. 当回退到根节点时，搜索结束。最后的“当前最近点”即为目标点x的最近邻点
#    5. 如果是找寻K个最近邻点，则需要比较target到当前节点父节点的的划分轴的距离，与target到其K个最近邻中距离最大的进行比较
#       如果前者小于等于后者，则说明在父节点的另一个子节点区域内很可能存在更近的最近邻，因此需要搜索。

# 如果实例点是随机分布的，kd树搜索的平均计算复杂度为O(logN)，这里N是训练实例数。kd树更适合于训练实例数远大于空间维数时的k近邻搜索。
# 当空间维数接近训练实例数时，它的效率会迅速下降，几乎接近线性扫描。

# 算法流程：
# 1. KD树的初始化
# 2. 按照某个维度划分子树
# 3. 按照划分依据计算中位数
# 4. 计算距离
# 5. 构建KD树
# 6. 搜索KD树
#================================================================================================================

from KDTree import KDTree
import numpy as np

class KNN_kdTree(KDTree):
    '''KNN算法实现：可用于分类，也可用于回归'''
    '''适用于批量预测'''
    def __init__(self,K):
        super(KNN_kdTree,self).__init__()

    def normalize(self,X, axis=-1, order=2):
        """ Normalize the dataset X  对各个特征向量进行归一化，消除量纲影响"""
        l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
        l2[l2 == 0] = 1 # 防止模长为0
        return X / np.expand_dims(l2, axis)

    def fit(self,X,Y):
        '''
        构建kd树
        X = np.array([[value,...],...])
        Y = np.array([label,...])
        label是非负整数
        '''
        X = self.normalize(X)
        Y = np.expand_dims(Y,axis=1)
        dataset = np.concatenate((X,Y),axis=1)
        super(KNN_kdTree,self).fit(dataset)

    def predict(self,target,K,isRegression=False):
        '''
        预测
        target = np.array([[value,...],...])
        K  个最近邻
        isRegression是否用于回归
        '''
        target = self.normalize(target)
        y_pred = [] # 存储预测结果
        k_data = [] # 存储每条数据的K个最近邻点
        for sample in target:
            result,data,label = super(KNN_kdTree,self).predict(sample,K,isRegression)
            y_pred.append(result)
            k_data.append(data)
        return np.array(y_pred),k_data

if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    # 计算准确率
    def accuracy_score(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
        return accuracy

    print('--------KNN KDTree Classification  --------')
    data = datasets.load_iris()
    X = data.data
    Y = data.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

    model = KNN_kdTree(K=5)
    model.fit(X_train,Y_train)
    y_pred,_ = model.predict(X_test,K=5,isRegression=False)
    accuracy = accuracy_score(Y_test, y_pred)
    print("Accuracy:", accuracy)