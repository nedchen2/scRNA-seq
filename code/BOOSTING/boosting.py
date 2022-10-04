#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score as ACCS
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier

#Open the original training and testing data
store_test = pd.HDFStore('C:/ftp/ml_10701_ps5_data.tar/test_data.h5')
store_train = pd.HDFStore('C:/ftp/ml_10701_ps5_data.tar/train_data.h5')
feature_matrix_dataframe_test = store_test['rpkm']
feature_matrix_dataframe_train =  store_train['rpkm']
labels_series_test = store_test['labels']
l_test = labels_series_test.values#label of testing set
labels_series_train = store_train['labels']
l_train= labels_series_train.values#label of the training set


new_train = pd.read_csv('new_train.csv')
new_test = pd.read_csv('new_test.csv')
pca = decomposition.PCA(n_components=50)
new_train_afterPCA = pca.fit_transform(new_train.values)
new_test_afterPCA = pca.transform(new_test.values)
# 降维
new_train_afterPCA_da = pd.DataFrame(new_train_afterPCA, index=new_train.index.values)
new_test_afterPCA_da = pd.DataFrame(new_test_afterPCA, index=new_test.index.values)



# In[ ]:


clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=100,learning_rate =1)
clf.fit(new_train_afterPCA_da, l_train)
pred_rfc = clf.predict(new_test_afterPCA_da)
score=ACCS(l_test, pred_rfc)
print(score)

