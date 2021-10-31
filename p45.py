import pandas
from sklearn import tree
import pandas as pd
import graphviz
from learning_lib import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv("pima-indians-diabetes.data.csv", names=names)
data.drop(columns=['age', 'test', 'skin', 'mass'], inplace=True)

train_data, test_data = train_test_split(data, test_size=0.15)
X_train, y_train = train_data.loc[:, "preg":"pedi"], train_data["class"]
X_test, y_test = test_data.loc[:, "preg":"pedi"], test_data["class"]

# decision tree

model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

print("decision tree")
print(model.score(X_test, y_test))


# random forest
model = RandomForestClassifier(n_estimators = 1000, random_state = 42)
model.fit(X_train, y_train)

y_predict = model.predict(X_test)
print("random forest")
print(model.score(X_test, y_test))
