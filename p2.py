import pandas
import copy
import numpy as np

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv("pima-indians-diabetes.data.csv", names=names)
continuous = {}
for i in ['plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age']:
    continuous[i] = {}
    continuous[i]["total_count"] = len(data[i])
    continuous[i]["cardinality"] = len(set(copy.deepcopy(data[i])))
    continuous[i]["range"] = [min(data[i]), max(data[i])]
    cnt = 0
    for j in data[i]:
        if np.isnan(j):
            cnt += 1
    continuous[i]["missing_count"] = cnt
print("continuous features")
for i in continuous.keys():
    print(continuous[i])
categorical = {}
for i in ['preg', 'class']:
    categorical[i] = {}
    categorical[i]["total_count"] = len(data[i])
    categorical[i]["cardinality"] = len(set(copy.deepcopy(data[i])))
    cnt = 0
    for j in data[i]:
        if np.isnan(j):
            cnt += 1
    categorical[i]["missing_count"] = cnt
print("categorical features")
for i in categorical.keys():
    print(categorical[i])
