#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import statements
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from sklearn.externals import joblib 


# In[2]:


#data input
df1 = pd.read_csv("/home/ghosh/Desktop/internships/shopclues101/modifieddata/newtrain.csv")
df = df1.where((pd.notnull(df1)), '')
pf1 = pd.read_csv("/home/ghosh/Desktop/internships/shopclues101/modifieddata/newtest.csv")
pf = pf1.where((pd.notnull(df1)), '')

#train set
# Categorize invalidity
df.loc[df["is_invalid"] == '1', "is_invalid",] = 1
df.loc[df["is_invalid"] == '0', "is_invalid",] = 0
# split data as label and address
df_x = df['address']
y_train = df['is_invalid']

#test set
# Categorize invalidity
pf.loc[df["is_invalid"] == '1', "is_invalid",] = 1
pf.loc[df["is_invalid"] == '0', "is_invalid",] = 0
# split data as label and address
pf_x = df['address']
y_test = df['is_invalid']


# In[3]:


# feature extraction, coversion to lower case and removal of stop words using TFIDF VECTORIZER
tfvec = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
x_trainFeat = tfvec.fit_transform(df_x)
x_testFeat = tfvec.fit_transform(pf_x)


# In[4]:


# SVM model
y_trainSvm = y_train.astype('int')
classifierModel = LinearSVC()
classifierModel.fit(x_trainFeat, y_trainSvm)
joblib.dump(classifierModel, '/home/ghosh/Desktop/internships/shopclues101/SVMmodel.pkl') 
predResult = classifierModel.predict(x_testFeat)


# In[5]:


# MNB model
y_trainGnb = y_train.astype('int')
classifierModel2 = MultinomialNB()
classifierModel2.fit(x_trainFeat, y_trainGnb)
joblib.dump(classifierModel2, '/home/ghosh/Desktop/internships/shopclues101/MNBmodel.pkl') 
predResult2 = classifierModel2.predict(x_testFeat)


# In[6]:


# Calc accuracy
y_test = y_test.astype('int')
actual_Y = y_test.to_numpy()

print("~~~~~~~~~~SVM RESULTS~~~~~~~~~~")
#Accuracy score using SVM
print("Accuracy Score using SVM: {0:.4f}".format(accuracy_score(actual_Y, predResult)*100))
#FScore MACRO using SVM
print("F Score using SVM: {0: .4f}".format(f1_score(actual_Y, predResult, average='macro')*100))
cmSVM=confusion_matrix(actual_Y, predResult)
#"[True negative  False Positive\nFalse Negative True Positive]"
print("Confusion matrix using SVM:")
print(cmSVM)

print("~~~~~~~~~~MNB RESULTS~~~~~~~~~~")
#Accuracy score using MNB
print("Accuracy Score using MNB: {0:.4f}".format(accuracy_score(actual_Y, predResult2)*100))
#FScore MACRO using MNB
print("F Score using MNB:{0: .4f}".format(f1_score(actual_Y, predResult2, average='macro')*100))
cmMNb=confusion_matrix(actual_Y, predResult2)
#"[True negative  False Positive\nFalse Negative True Positive]"
print("Confusion matrix using MNB:")
print(cmMNb)

