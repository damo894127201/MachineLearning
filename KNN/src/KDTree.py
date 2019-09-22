# -*- coding: utf-8 -*-
# @Time    : 2019/9/21 8:11
# @Author  : Weiyang
# @File    : KDTree.py

#======================================================
# KDTree
# 构建KD树和搜索KD树
# 使用于一条数据一条数据的预测
#======================================================

from KDNode import KDNode
import numpy as np

class KDTree:
    '''构建KD树和搜索KD树'''
    def __init__(self):
        self.root = None # 根节点
        self.dimensions = None # 记录维度数，总共有多少个特征
        self.k_neighbour = [] # 记录最近的K个节点 ，空列表便于创建堆
        self.path = [] # 记录搜索路径

    def _getAxis(self,depth,n_features):
        '''
        根据树的深度，确定划分轴的索引，注意轴从0开始编号
        depth 树的深度，根节点的深度为0
        n_features: 训练实例的特征数，维度数，从0开始编号
        '''
        return depth % n_features

    def _getSortDataset(self,dataset,axis):
        '''
        根据指定列(特征的指定维度)，对数据进行排序
        dataset: 训练实例数据集，可能是全量集，也可能是子集；axis：划分轴的编号,[0,1,2,...]
        '''
        sort_index = np.argsort(dataset,axis=0)
        return dataset[sort_index[:,axis]] # 返回按指定列排序的数据集

    def _getDist(self,current_node,target):
        '''
        计算当前节点与目标节点之间的距离
        current_node: 当前节点
        target：目标点的特征值向量,np.array([value,...])
        '''
        # 欧式距离
        distance = (((current_node.feature_value - target) ** 2).sum()) ** 0.5
        return distance

    def createKDTree(self,dataset,depth=0,father_node=None):
        '''
        创建KD树：
        dataset ,含label的数据集：np.array([[value,...,lable],...])
        depth: 当前节点的深度
        father_node: 当前节点的父节点

        流程：
        1. 如果数据集中只有一条数据，则用该数据创建叶子结点
        2. 如果不止一条数据，则进行如下操作：
           1. 根据kd树当前的深度，选定划分轴
           2. 根据划分轴，对数据集按照该特征从小到大排序
           3. 选出中位数，确定划分值(该轴数据的中位数)
           4. 小于划分值的数据划到左节点，大于划分值的数据划到右节点
           5. 递归调用自身，构造KDTree
        '''
        n_samples = dataset.shape[0] # 数据集个数
        # 判断数据集个数
        if n_samples < 1:
            return None
        elif n_samples == 1:
            # 创建叶子节点
            current_node = KDNode(feature_value=dataset[0,:-1],label=dataset[0,-1],depth=depth)
            current_node.father = father_node # 设定该叶节点的父节点
        # 继续划分数据集
        else:
            # 获取当前节点的划分轴编号
            split_axis = self._getAxis(depth,self.dimensions)
            # 对数据集按划分轴split_axis排序
            sorted_dataset = self._getSortDataset(dataset,split_axis)
            # 获取划分轴的中位数，对应的数据索引
            media_index = n_samples // 2

            # 构造当前节点
            current_node = KDNode(feature_value=sorted_dataset[media_index,:-1],
                                  label=sorted_dataset[media_index,-1],split_axis=split_axis,depth=depth)

            # 构造当前节点的左子节点
            # 获取分割到左子节点的数据集
            left_data = sorted_dataset[:media_index]
            # 递归创建树
            current_node.left_branch = self.createKDTree(left_data,depth+1,current_node)

            # 构造当前节点的右子节点
            # 获取分割到右子节点的数据集
            right_data = sorted_dataset[media_index+1:]
            # 递归创建树
            current_node.right_branch = self.createKDTree(right_data,depth+1,current_node)

            # 当前节点的父节点
            current_node.father = father_node

        return current_node # 返回当前节点

    def getKDTreeDepth(self,node):
        '''node 是节点，返回以该节点为根节点的树的深度'''
        if node is None:
            return 0
        else:
            return max(self.getKDTreeDepth(node.left_branch),
                       self.getKDTreeDepth(node.right_branch)) + 1 # 妙!

    def fit(self,dataset):
        '''
        用训练数据构建kd树
        dataset ,含label的数据集：np.array([[value,...,lable],...])
        label: 必须是非负整数
        '''
        self.dimensions = len(dataset[0]) - 1 # 存储特征数
        self.root = self.createKDTree(dataset) # 构建kd树
        depth = self.getKDTreeDepth(self.root) # 获取树的深度
        print("The KD Tree's depth is ",depth )

    def _searchLeafNode(self,node,target):
        '''
        找出包含目标点的叶节点或中间节点，中间节点的含义是指 target无法分配到叶节点内，只能分配到叶节点的父节点上
        node: 节点 ； target ：目标点
        '''
        # 如果当前节点是叶节点,则返回当前叶节点
        if node.left_branch == None and node.right_branch == None:
            return node
        # 获取当前节点的划分轴编码
        split_axis = node.split_axis
        # 确定当前节点是在左节点所在的区域，还是右节点所在的区域
        if target[split_axis] < node.feature_value[split_axis] and node.left_branch != None:
            # 往左节点走
            return self._searchLeafNode(node.left_branch,target)

        if target[split_axis] >= node.feature_value[split_axis] and node.right_branch != None:
            # 往右节点走
            return self._searchLeafNode(node.right_branch,target)
        # 当数据集只剩两个实例时，由于取中位数对应实例的索引是n_samples//2,因此会选取索引为1的实例为切割点
        # 也就是说只会生成左子节点，而不会生成右子节点
        # 当右节点为空时，返回当前节点，此时返回的不是叶节点，而是叶节点的父节点
        elif target[split_axis] >= node.feature_value[split_axis] and node.right_branch == None:
            return node

    def _searchKDTree(self,current_node,target,K):
        '''递归回退，搜索KDTree'''
        if current_node is None:
            return
        # 如果该节点未被访问过，则计算其与target的距离
        if current_node not in self.path:
            # 计算当前节点current_node与target的距离
            distance = self._getDist(current_node,target)
        else:
            return  # 如果节点被访问过，则停止继续向下探索
        # 如果还未找足K个最近邻，且该节点未被访问过，则将当前节点加入到K近邻列表中
        if len(self.k_neighbour) < K and current_node not in self.path:
            self.k_neighbour.append({"node":current_node,"distance":distance})
            # 将当前节点加入到搜索路径中
            self.path.append(current_node)
        # 如果K个最近邻列表中已满，且当前节点未被访问过，则比较当前节点与target的距离distance ,与self.k_neighbour中距离最大
        # 的值大小,如果distance较小，则用distance替换掉对应距离，用当前节点替换掉相应的节点
        elif len(self.k_neighbour) == K and current_node not in self.path:
            # 将当前节点加入到搜索路径中
            self.path.append(current_node)
            # 对self.k_neighbour按照距离排序，找出distance最大的节点其在self.k_neighbour的索引
            self.k_neighbour = sorted(self.k_neighbour,key=lambda i:i['distance'])
            # 如果距离更小，则进行替换
            if distance < self.k_neighbour[-1]['distance']:
                self.k_neighbour[-1] = {"node":current_node,"distance":distance}

        # 在当前节点的左右子节点中查询k近邻点
        # 访问左节点
        self._searchKDTree(current_node.left_branch,target,K)
        # 访问右节点
        self._searchKDTree(current_node.right_branch,target,K)

        # 对距离进行排序
        self.k_neighbour = sorted(self.k_neighbour, key=lambda i: i['distance'])

        # 判断self.k_neighbour的点是否已满，
        # 如果已满，则判断target到当前节点的父节点 划分轴的距离是否小于self.k_neighbour的最大距离
        # 如果小于，则说明在当前节点的兄弟节点及其子区域中很可能存在更近的点
        # target到父节点的划分轴的距离为 target在相应划分轴的值减去父节点划分轴上的值 的 绝对值

        # 判断当前节点是否为根节点
        if current_node != self.root:
            # 当前节点的父节点
            father = current_node.father
            dis = abs(target[father.split_axis] - father.feature_value[father.split_axis])
            if len(self.k_neighbour) == K and self.k_neighbour[-1]['distance'] >= dis:
                # 继续向上回退到父节点
                # 判断父节点是否访问过，如果访问过，就不再继续向下探索
                if current_node.father not in self.path:
                    self._searchKDTree(current_node.father, target, K)
            elif len(self.k_neighbour) == K and self.k_neighbour[-1]['distance'] < dis:
                # 说明不存在更近的点了，退出搜索
                return
                # 如果最近邻点还不足，继续往父节点搜索
            if len(self.k_neighbour) < K:
                # 判断父节点是否为空，以及是否访问过，如果访问过，就不再继续向下探索
                if current_node.father not in self.path and current_node.father != None:
                    self._searchKDTree(current_node.father, target, K)
        else:
            # 如果当前节点为根节点，则退出搜索，因为上面已经搜索过左右子节点了
            return

    def searchKDTree(self,target,K):
        '''
        搜索KD树，寻找target的最近邻的K个点
        target: 搜索目标
        K：最近邻点的个数
        '''
        if K < 1:
            raise ValueError("K 必须大于0 ！")
        if self.root is None:
            raise ValueError("KD树 是空的！")
        if len(target) != self.dimensions:
            raise ValueError("target 的维度与训练实例的维度不一致")

        # 找寻包含target的叶节点 或 叶节点的父节点
        leaf_node = self._searchLeafNode(self.root,target)
        # 将叶节点加入到k个最近邻列表中，并记录其与target的距离
        distance = self._getDist(leaf_node,target)
        self.k_neighbour.append({"node":leaf_node,"distance":distance})
        # 将当前节点加入到搜索路径中，表明该节点已访问过
        self.path.append(leaf_node)

        # 判断包含target的是叶节点，还是叶节点的父节点
        if leaf_node.left_branch != None:
            # 将该节点的左子节点加入到k个最近邻列表中，并记录其与target的距离
            distance = self._getDist(leaf_node.left_branch, target)
            self.k_neighbour.append({"node": leaf_node.left_branch, "distance": distance})
            # 将当前节点加入到搜索路径中，表明该节点已访问过
            self.path.append(leaf_node.left_branch)

        # 从包含target的叶节点 或 叶节点的父节点 开始，向上回退到父节点，寻找距离target最近的k个样本点
        self._searchKDTree(leaf_node.father,target,K)

    def predict(self,target,K=3,isRegression=False):
        '''
        预测当前节点的label或y值，同时输出当前节点最近邻的K个节点
        target: np.array([value,..value]) 不含label或值
        '''
        # 搜索KD树
        self.searchKDTree(target,K)
        # 获取最近邻的K个数据点的特征向量
        data = [node["node"].feature_value for node in self.k_neighbour]
        # 获取最近邻的K个数据点的label或y值
        label = [node["node"].label for node in self.k_neighbour]
        if isRegression:
            predict_value = np.array(label).mean()
            self.k_neighbour = [] # 清空，便于下次预测
            self.path = [] # 清除扫描路径
            return predict_value,data,label
        else:
            most_label = label[np.argmax(np.array(label))]
            self.k_neighbour = []  # 清空，便于下次预测
            self.path = [] # 清除扫描路径
            # 返回多数类
            return most_label,data,label

if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    # 计算准确率
    def accuracy_score(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
        return accuracy


    def normalize(X, axis=-1, order=2):
        """ Normalize the dataset X """
        l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
        l2[l2 == 0] = 1 # 防止模长为0
        return X / np.expand_dims(l2, axis)

    print('--------KNN KDTree Classification  --------')
    data = datasets.load_iris()
    X = normalize(data.data) # 对数据进行归一化，去除各个特征间的量纲影响
    Y = data.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

    Y_train = np.expand_dims(Y_train,axis=1)
    # 将 X_train,Y_train 拼接到一起
    train_dataset = np.concatenate((X_train,Y_train),axis=1)
    model = KDTree()
    model.fit(train_dataset)

    y_pred = []
    for sample in X_test:
        result,data,label = model.predict(sample,K=5) # 返回预测结果,相应的最近邻点,最近邻点的label
        y_pred.append(result)
    y_pred = np.array(y_pred)
    accuracy = accuracy_score(Y_test, y_pred)
    print("Accuracy:", accuracy)