# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:31:56 2020

@author: dolapo
"""

#importing libraries
import timeit #import timeit for runtime tracking
import numpy as np #numpy is use for slicing numpy Array
import matplotlib.pyplot as plt #matplot library is used for plotting 
import pandas as pd #Pandas is use for data import and manipulation 

start = timeit.default_timer() #timer start here 
#importing the database
data=pd.read_excel('db1.xlsx')
index=data['Date']
#set the date as index
data.index=pd.to_datetime(data.index)
#data.head()
#split the data into input and output 
inputs=['Rainfall','TempMin','TempMax']
outputs=['TARGET']
#the target/output is converted to integer
target={'Blackpod':1,'No Blackpod':0}
data=data.replace({'TARGET':target})
#X.head()

#splitting the data into train and test set
from sklearn.model_selection import train_test_split #libarary use for splitting data into train and test data 
X_train,X_test,y_train,y_test=train_test_split(data[inputs],data[outputs],test_size=0.25,random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler #library for standard scaling of data
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#fitting classifier to the training set
from sklearn.tree import DecisionTreeClassifier #import decision tree classifier 
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

#prediction
y_pred=classifier.predict(X_test)

stop = timeit.default_timer() #end of execution time 

print('Time: ', stop - start)  #calculate the time 

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