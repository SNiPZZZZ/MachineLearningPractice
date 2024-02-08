import pandas as pd

dataset = pd.read_csv('BreastCancerUdemy//breast_cancer.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(xTrain, yTrain)

yPred = classifier.predict(xTest)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(yTest, yPred)
print(accuracy_score(yTest, yPred))
print(cm)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(classifier,xTrain,yTrain,cv=10)
print("Accuracy: {:.2f} % ".format(accuracies.mean()*100))
print("Standard deviation: {:.2f} % ".format(accuracies.std()*100))
