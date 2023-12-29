import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

dataset = pd.read_csv('2_Regression\\Datasets\\3_Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y), 1)

from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
sc2 = StandardScaler()
x = sc1.fit_transform(x)
y = sc2.fit_transform(y)

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x,y)

print(sc2.inverse_transform(regressor.predict(sc1.transform([[6.5]])).reshape(-1,1)))

plt.scatter(sc1.inverse_transform(x),sc2.inverse_transform(y),color='red')
plt.plot(sc1.inverse_transform(x),sc2.inverse_transform(regressor.predict(x).reshape(-1,1)),color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()






