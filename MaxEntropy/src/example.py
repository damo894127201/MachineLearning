# -*- coding: utf-8 -*-
# @Time    : 2019/9/30 12:21
# @Author  : Weiyang
# @File    : example.py

import pandas as pd
import numpy as np
from MaxEntropy import MaxEntropy

# 构造数据
raw_data =np.array([['no','sunny','hot','high','FALSE'],
    ['no','sunny','hot','high','TRUE'],
    ['yes','overcast','hot','high','FALSE'],
    ['yes','rainy','mild','high','FALSE'],
    ['yes','rainy','cool','normal','FALSE'],
    ['no','rainy','cool','normal','TRUE'],
    ['yes','overcast','cool','normal','TRUE'],
    ['no','sunny','mild','high','FALSE'],
    ['yes','sunny','cool','normal','FALSE'],
    ['yes','rainy','mild','normal','FALSE'],
    ['yes','sunny','mild','normal','TRUE'],
    ['yes','overcast','mild','high','TRUE'],
    ['yes','overcast','hot','normal','FALSE'],
    ['no','rainy','mild','high','TRUE']])

data = {'play':raw_data[:,0], 'outlook':raw_data[:,1], 'temperature':raw_data[:,2],
                    'humidity':raw_data[:,3], 'windy':raw_data[:,4]}
data = pd.DataFrame(data)
#print(data)
# 我们用第一列作为label，即是否出去 play为标签
#print(raw_data[:, 1:])
X = raw_data[:, 1:]
Y = raw_data[:,0]
model = MaxEntropy(X,Y)
model.train()
X_test = [['sunny','hot','high','FALSE'],
          ['overcast',None,'high','FALSE'], # 有缺失值
          ['sunny','cool','high','TRUE']]
y_pred,all_pred = model.predict(X_test)
for pred in all_pred:
   print(pred)