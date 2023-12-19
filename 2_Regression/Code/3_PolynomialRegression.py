import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

dataset = pd.read_csv('2_Regression\\Datasets\\3_Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
LinReg = LinearRegression()
LinReg.fit(x,y)

from sklearn.preprocessing import PolynomialFeatures
PolyReg = PolynomialFeatures(degree = 4)
xPoly = PolyReg.fit_transform(x)
LinReg2 = LinearRegression()
LinReg2.fit(xPoly,y)

plt.scatter(x,y,color='red')
plt.plot(x,LinReg.predict(x),color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

xGrid = np.arange(min(x),max(x),0.1)
xGrid = xGrid.reshape((len(xGrid),1))
xPoly = PolyReg.fit_transform(x)
plt.scatter(x,y,color='red')
plt.plot(x,LinReg2.predict(xPoly),color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

print(LinReg.predict([[6.5]]))

print(LinReg2.predict(PolyReg.fit_transform([[6.5]])))

