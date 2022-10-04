import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.feature_selection import VarianceThreshold #导入python的相关模块

# open up a datastore
store = pd.HDFStore('train_data.h5')
store1 = pd.HDFStore('test_data.h5')
# Get the feature matrix (samples and their features)
feature_matrix_dataframe = store['rpkm']     #train的数据框架
feature_matrix_dataframe_test = store1['rpkm']   #test的数据框架

all_in = pd.concat([feature_matrix_dataframe,feature_matrix_dataframe_test],axis=0,join='inner')

#清理train数据集
name_all=feature_matrix_dataframe.columns.values #提取gene标签
name_select=[]

for item in name_all:       
    if feature_matrix_dataframe[item].var()< 15:  #如果方差小于10
        name_select.append(item)  #记录

for column in name_select:      #挨个删除列
    del feature_matrix_dataframe[column]

all_in = pd.concat([feature_matrix_dataframe,feature_matrix_dataframe_test],axis=0,join='inner')  #合起来，删掉test中不需要的feature
new_train = all_in.iloc[:21389,]  #新的train数据 
new_test = all_in.iloc[21389: ,]  #新的test数据

