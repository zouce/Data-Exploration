import pandas
import numpy as np

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv("pima-indians-diabetes.data.csv", names=names)
for ii in range(len(names)):
    i = names[ii]
    for jj in range(ii+1, len(names)):
        j = names[jj]
        if i == j or i == 'class' or j == 'class':
            continue
        mediana = np.mean(data[i])
        medianb = np.mean(data[j])
        cov = 0
        tt = 0
        for k in range(len(data[i])):
            if (data[i][k] == 0 and i != 'preg') or (data[j][k] == 0 and j != 'preg'):
                tt += 1
                continue
            cov += (data[i][k] - mediana) * (data[j][k] - medianb)
        cov /= len(data[i]) - tt
        stda = np.std(data[i])
        stdb = np.std(data[j])
        corr = cov/(stda*stdb)
        # print(i, j, corr)
        if corr >= 0.4:
            print(i, j, corr)
