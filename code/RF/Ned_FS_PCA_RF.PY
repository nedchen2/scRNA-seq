import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.feature_selection import VarianceThreshold #导入python的相关模块
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score as ACCS
import pandas as pd
from sklearn import tree
from sklearn import decomposition
from sklearn.model_selection import ShuffleSplit

# open up a datastore
store = pd.HDFStore('train_data.h5')
store1 = pd.HDFStore('test_data.h5')
# Get the feature matrix (samples and their features)
feature_matrix_dataframe = store['rpkm']     #train的数据框架
feature_matrix_dataframe_test = store1['rpkm']   #test的数据框架

all_in = pd.concat([feature_matrix_dataframe,feature_matrix_dataframe_test],axis=0,join='inner')

name=all_in.index.values   #提取行标签

data=all_in.values  #提取数据丢掉标签（总和数据）


sel=VarianceThreshold(threshold=15)  #表示剔除特征的方差小于阈值 i的feature
new=sel.fit_transform(data)#返回的结果为选择的特征矩阵
new_all=pd.DataFrame(new,index=name) #得到只剩下feature的新数据
a=new_all.iloc[:5,3]
new_train = new_all.iloc[:21389,]  #新的train数据 
new_test = new_all.iloc[21389: ,]  #新的test数据

    #注意，新数据的gene序列（columns）被抹掉了，但是这不重要
    #唯一的问题是所有的数据一起提取方差了

pca = decomposition.PCA(n_components = 'mle')  #这边可以用MLE吗？
new_train_afterPCA = pca.fit_transform(new_train.values)
new_test_afterPCA = pca.transform(new_test.values)
            #降维
new_train_afterPCA_da = pd.DataFrame(new_train_afterPCA, index=new_train.index.values, columns=new_train.cloumns.values )
new_test_afterPCA_da = pd.DataFrame(new_test_afterPCA, index=new_test.index.values, columns=new_test.cloumns.values )

labels_series1 = store1['labels']
test_data_lab = labels_series1.values         #test data
labels_series = store['labels']
train_data_lab = labels_series.values      #train data

clf = RandomForestClassifier(min_samples_leaf=55, n_estimators=150, max_depth=30, n_jobs=2, random_state=0)
clf.fit(new_train_afterPCA_da, train_data_lab)
# print(new_test_afterPCA_da)
# print(type(new_test_afterPCA_da))
# print(new_test_afterPCA_da.shape)

pred_rfc = clf.predict(new_test_afterPCA_da)

# print(pred_rfc)
# print(type(pred_rfc))
# print(pred_rfc.shape)

score=ACCS(test_data_lab, pred_rfc)
print("Accuracy")
print(score)

CV = cross_val_score(clf, new_test.rpkm, new_test.label, cv= 10)
Print ("CV")
print (CV)