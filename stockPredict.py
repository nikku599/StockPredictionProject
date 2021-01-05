# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 21:49:38 2021

@author: nikhil sharma
"""




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the data
df=pd.read_csv("NSE-TATAGLOBAL11.csv")
df.info()

#Another important thing to note is that the market is closed on 
#weekends and public holidays.Notice the above table again,
#some date values are missing – 2/10/2018, 6/10/2018, 7/10/2018. 
#Of these dates, 2nd is a national holiday while 6th and 7th fall on a 
#weekend.

df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

#plotting the data
plt.figure(figsize=(16,8))
plt.plot(df['Close'], label='Close Price history')

#sorting
data = df.sort_index(ascending=True, axis=0)

#feature selection into new dataset
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

#creating features
new_data["Year"]=new_data.Date.dt.year
new_data["Month"]=new_data.Date.dt.month
new_data["Week"]=new_data.Date.dt.week
new_data["Day"]=new_data.Date.dt.day
new_data["Dayofweek"]=new_data.Date.dt.dayofweek
new_data["is_month_end"]=new_data.Date.dt.is_month_end
new_data["is_quarter_end"]=new_data.Date.dt.is_quarter_end
new_data["is_quarter_start"]=new_data.Date.dt.is_quarter_start
new_data["is_year_end"]=new_data.Date.dt.is_year_end
new_data["is_year_start"]=new_data.Date.dt.is_year_start

#weekdays afftects prices more often in share market
new_data['mon_fri'] = 0
for i in range(0,len(new_data)):
    if (new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4):
        new_data['mon_fri'][i] = 1
    else:
        new_data['mon_fri'][i] = 0
        
X=new_data.drop(["Close","Date"],axis=1)
Y=new_data["Close"]
Y=pd.to_numeric(Y)

#spliting into training and testing dataset
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.25)

#Training new_data with linearRegresion model
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(xtrain,ytrain)

#predicting values
y_pred=model.predict(xtest)

arr=ytest.to_list()

#showing the predicted value and actual values thru graphs of test_cases
plt.figure(figsize=(16,8))
plt.plot(arr)
plt.plot(y_pred,color="red")

#finally predicting the values of complete dataset and plotting by graphs
y_pred_final=model.predict(X)

plt.figure(figsize=(16,8))
plt.plot(Y, label='Linear Regression plot')
plt.plot(y_pred_final,color="red")

#model accuracy
from sklearn.metrics import r2_score
print(r2_score(Y,y_pred_final))

#model root mean error
from sklearn.metrics import mean_squared_error
print(mean_squared_error(Y, y_pred_final))
