import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

dataset = pd.read_csv('2_Regression\\Datasets\\2_50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

from sklearn.model_selection import train_test_split
xTrain,xTest,yTrain,yTest = train_test_split(x, y, test_size = 0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xTrain,yTrain)

yPred = regressor.predict(xTest)
np.set_printoptions(precision= 2)
print(np.concatenate((yPred.reshape(len(yPred),1), yTest.reshape(len(yTest),1)), 1))

