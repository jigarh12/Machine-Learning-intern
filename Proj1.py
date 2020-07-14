# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 14:42:35 2020

@author: Jigar
"""
#Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import Files

data=pd.read_csv('advertising.csv')

data.head()


#To visualise data

fig , axis = plt.subplots(1,3,sharey= True)
data.plot(kind='scatter',x='TV',y='Sales',ax=axis[0],figsize=(14,7))
data.plot(kind='scatter',x='Radio',y='Sales',ax=axis[1],figsize=(14,7))
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axis[2],figsize=(14,7))


#Creating X&Y for Linear Regression
feature_cols=['TV']
x=data[feature_cols]
y=data.Sales


#importing Linear Regression Algo
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x,y)

print(lr.intercept_)
print(lr.coef_)

#y= a+b*x

result=6.974821 + 0.055464*50
print(result)

#create a dataframe with Min and Max value of the table
x_new = pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
x_new.head()


pred=lr.predict(x_new)
print(pred)

data.plot(kind='scatter',x='TV',y='Sales')
plt.plot(x_new,pred,color='red',linewidth=3)


import statsmodels.formula.api as smf
lr=smf.ols(formula='Sales ~ TV',data=data).fit()
lr.conf_int()


#finding the Probablity Value
lr.pvalues

#Finding the R-Sqaured value
lr.rsquared

#Multi Linear Regression
feature_cols=['TV','Radio','Newspaper']
x=data[feature_cols]
y=data.Sales

lr=LinearRegression()
lr.fit(x,y)

print(lr.intercept_)
print(lr.coef_)

lm=smf.ols(formula='Sales~TV+Radio+Newspaper',data=data).fit()
lm.conf_int()
lm.summary()
