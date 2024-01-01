import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

dataset = pd.read_csv('2_Regression\\Datasets\\3_Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x, y)

print(regressor.predict([[6.5]]))

xGrid = np.arange(min(x),max(x),0.1)
xGrid = xGrid.reshape((len(xGrid),1))
plt.scatter(x,y,color='red')
plt.plot(xGrid,regressor.predict(xGrid),color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()