import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

dataset = pd.read_csv("C:\\Users\\USP\\Desktop\\ML\\Machine Learning A-Z (Codes and Datasets)\\Part 2 - Regression\\Section 4 - Simple Linear Regression\\Python\\Salary_Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
xTrain,xTest,yTrain,yTest = train_test_split(x,y,test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xTrain, yTrain)

yPred = regressor.predict(xTest)

plt.scatter(xTrain, yTrain, color = 'red')
plt.plot(xTrain, regressor.predict(xTrain), color= 'blue')
plt.title('Salary vs Experience (training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(xTest, yTest, color = 'red')
plt.plot(xTrain, regressor.predict(xTrain), color= 'blue')
plt.title('Salary vs Experience (test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

print(regressor.predict([[1000]]))

print(regressor.coef_)
print(regressor.intercept_)