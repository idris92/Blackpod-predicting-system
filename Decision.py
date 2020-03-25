# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:31:56 2020

@author: dolapo
"""

#importing libraries
import timeit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

start = timeit.default_timer()
#importing the database
data=pd.read_excel('db1.xlsx')
index=data['Date']
data.index=pd.to_datetime(data.index)
#data1={'Blackpod':1,'No Blackpod':0}
#data=data.replace({'TARGET':data1})
#data.head()
inputs=['Rainfall','TempMin','TempMax']
outputs=['TARGET']
target={'Blackpod':1,'No Blackpod':0}
data=data.replace({'TARGET':target})
#X.head()

#splitting the data into train and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data[inputs],data[outputs],test_size=0.25,random_state=0)

#Feature scling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#fitting classifier to the training set
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

#prediction
y_pred=classifier.predict(X_test)

stop = timeit.default_timer()

print('Time: ', stop - start)  

#confusion metrics
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
#mean squared error
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(y_test,y_pred))
#preccision,recall,accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)