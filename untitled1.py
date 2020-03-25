# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 07:01:47 2020

@author: dolapo
"""

import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pmdarima.arima import auto_arima

dataset=pd.read_excel('db1.xlsx')
dataset=dataset.set_index('Date')
stepwise_model=auto_arima(dataset['TempMax'], start_p=1, start_q=1,
                              max_p=5, max_q=5, m=12,
                              start_P=0, seasonal=True,
                              d=1, D=1, trace=True, error_action='ignore',
                              suppress_warnings=True, stepwise=True)
stepwise_model.fit(dataset['TempMax'])
forecast=stepwise_model.predict(n_periods=len(dataset['TempMax']))
forecast = pd.DataFrame(forecast,columns=['Prediction'])
new_data=dataset.join(forecast.set_index(dataset.index))

inputs=['Rainfall','TempMin','Prediction']
outputs=['TARGET']
X_train,X_test,y_train,y_test=train_test_split(new_data[inputs],new_data[outputs],test_size=0.25,random_state=0)
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

import pickle
filename='model1.pkl'
with open('C:/Users/dolapo/Documents/BlackPod/'+filename,'wb')as file:
    pickle.dump(classifier,file)

with open('C:/Users/dolapo/Documents/BlackPod/model1.pkl','rb')as f:
    loaded_model=pickle.load(f)
import pandas as pd
my_data=pd.DataFrame({'Rainfall':[0.7],'TempMin':[18.6],'TempMax':[33]})
loaded_model.predict(my_data)