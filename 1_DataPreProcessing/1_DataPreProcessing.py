import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

dataset = pd.read_csv('C:\\Users\\USP\\Desktop\\ML\\Machine Learning A-Z (Codes and Datasets)\\Part 1 - Data Preprocessing\\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\\Python\\Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

#print(x)
#print(y)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#print(x)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

#print(x)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y) 

#print(y)

from sklearn.model_selection import train_test_split
xTrain,xTest,yTrain,yTest = train_test_split(x, y, test_size = 0.2, random_state=1)

#print(xTrain)
#print(xTest)
#print(yTrain)
#print(yTest)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xTrain[:, 3:] = sc.fit_transform(xTrain[:, 3:])
xTest[:, 3:] = sc.transform(xTest[:, 3:])

#print(xTrain)
#print(xTest)

