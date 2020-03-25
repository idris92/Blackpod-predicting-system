# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 19:16:48 2020

@author: dolapo
"""
#import part
import timeit
import pandas as pd
from pmdarima.arima import auto_arima
import matplotlib.pyplot as plt

start = timeit.default_timer()
#loading the dataset
dataset=pd.read_excel('db1.xlsx')
dataset.head()

#preprocessing stage
index=['Date']
dataset=dataset.set_index(index)
drops=['Rainfall','TempMin','TARGET']
dataset.drop(drops,axis=1,inplace=True)
#output={'Blackpod':1,'No Blackpod':0}
#dataset=dataset.replace({'TARGET':output})
#dataset.head()
#divide into test and train set
##train = dataset[:int(0.75*(len(dataset)))]
##test = dataset[:int(0.25*(len(dataset))):]


#plotting the data
##plt.plot(train,label='Train')
##plt.show()


#building the model
stepwise_model=auto_arima(dataset, start_p=1, start_q=1,
                          max_p=5, max_q=5, m=12,
                          start_P=0, seasonal=True,
                          d=1, D=1, trace=True, error_action='ignore',
                          suppress_warnings=True, stepwise=True)

print(stepwise_model.aic())
#traintest
train=dataset.loc['1988-01-01':'2010-12-01']
test=dataset.loc['2011-01-01':]

stepwise_model.fit(train)
forecast=stepwise_model.predict(n_periods=len(test))

stop = timeit.default_timer()

print('Time: ', stop - start)  

#forecast = pd.DataFrame(forecast,index = test.index,columns=['Prediction'])


#confusion matrix
#pred=forecast['Prediction']
new_list=[]
for item in forecast:
    if item<=30:
        new_list.append(1)
    else:
        new_list.append(0)
#print(new_list)
tests=test['TempMax']
new_list1=[]
for item in tests:
    if item<=30:
        new_list1.append(1)
    else:
        new_list1.append(0)

#Root mean square
from sklearn.metrics import mean_squared_error
from math import sqrt
rmss=sqrt(mean_squared_error(new_list1,new_list))
       
#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(new_list,new_list1)
#plot the prediction 
plt.plot(train, label='Train')
plt.plot(test, label='test')
plt.plot(forecast, label='Prediction')
plt.show()
#calculate rmse
from math import sqrt
from sklearn.metrics import mean_squared_error

rms = sqrt(mean_squared_error(test,forecast))
print(rms)
#precision,recll,accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
accuracy = accuracy_score(new_list,new_list1)
precision = precision_score(new_list1,new_list)
recall = recall_score(new_list,new_list1)