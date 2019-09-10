# -*- coding: utf-8 -*-
# @Time    : 2019/6/29 14:07
# @Author  : Weiyang
# @File    : compare.py

########################################################################################################################
# 用于比较编写的算法与sklearn.cluster.KMeans 算法
# 数据集：西瓜数据集和自定义数据集
########################################################################################################################

from sklearn import metrics
import pandas as pd
import random

# ----------------------------  导入西瓜数据 -----------------------
data = pd.read_csv('../data/watermelon4.0.csv',encoding='utf-8',index_col=0)
# 获取数据点:[[value1,value2],[value3,value4],...]
points1 = data.values.tolist()

# -------------------------- 自定义数据集：生成一个样本集，用于测试KMmeans算法 ------------
def get_test_data():
    N = 1000
    # 产生点的区域,共5个区域，会聚类成5个点簇
    area_1 = [0, N / 4, N / 4, N / 2]
    area_2 = [N / 2, 3 * N / 4, 0, N / 4]
    area_3 = [N / 4, N / 2, N / 2, 3 * N / 4]
    area_4 = [3 * N / 4, N, 3 * N / 4, N]
    area_5 = [3 * N / 4, N, N / 4, N / 2]
    areas = [area_1, area_2, area_3, area_4, area_5]
    # 在各个区域内，随机产生一些点
    points = []
    for area in areas:
        rnd_num_of_points = random.randint(50, 200)
        for r in range(0, rnd_num_of_points):
            rnd_add = random.randint(0, 100)
            rnd_x = random.randint(area[0] + rnd_add, area[1] - rnd_add)
            rnd_y = random.randint(area[2], area[3] - rnd_add)
            points.append([rnd_x, rnd_y])
    return points
# 生成测试集
points2 = get_test_data()

# -----------------------------------------------------------------
points = [points1,points2]

count = 0
for point in points:
    # 方法一
    '''
    from Kmeans_distance import KMeans
    k = KMeans(3, distance='E')
    labels = k.fit_predict(point)
    result = metrics.silhouette_score(point,labels,metric='euclidean')
    print(count,'Kmeans_distance: ',round(result,2))

    # 方法二
    from Kmeans_cosine import KMeans
    k = KMeans(3)
    labels = k.fit_predict(point)
    result = metrics.silhouette_score(point,labels,metric='euclidean')
    print(count,'Kmeans_cosine: ',round(result,2))
    '''

    # 方法三
    from sklearn.cluster import KMeans
    k = KMeans(n_clusters=3)
    k.fit_predict(point)
    labels = k.labels_
    result = metrics.silhouette_score(point,labels,metric='euclidean')
    print(count,'Sklearn.cluster.KMeans: ',round(result,2))
    print()
    count += 1