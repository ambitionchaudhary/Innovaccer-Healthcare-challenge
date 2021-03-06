# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 15:52:02 2018

@author: Ambition
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = (20.0, 10.0)
import seaborn as sns
from scipy import stats
from scipy.stats import norm
##import graphviz
from sklearn import preprocessing,model_selection
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

 df=pd.read_csv("file:///C:/Users/Ambition/Documents/Data cases/Innovacer/final.csv")
df.info()
df.columns
df.describe()
df.fillna(df.median(),inplace=True)

#Correlation matrix
corr = df.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)

 #Linear regression
X = df.drop('per_capita_exp_total_py', axis = 1)
Y = df['per_capita_exp_total_py']
regr = LinearRegression()
rfr =  RandomForestRegressor()
gbm = GradientBoostingRegressor()
gbm.fit(X,Y)
rfr.fit(X,Y)
regr.fit(X,Y)
y_pred = regr.predict(X)
y_rf_pred=rfr.predict(X)
# The coefficients
print('Coefficients: \n', regr.coef_)
print('Coefficients: \n',regr.intercept_)

#Feature importance


feature_importances = pd.DataFrame(rfr.feature_importances_,index = X_train.columns,columns=['importance']).sort_values('importance', ascending=False)


