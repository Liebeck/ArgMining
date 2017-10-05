from sklearn.metrics import f1_score
import numpy as np
from pandas_confusion import ConfusionMatrix

with open('C_baseline.predictions') as file:
    y_true = []
    y_pred = []
    next(file)
    for line in file:
        row = line.replace('\n', '').split('\t')
        y_true.append(row[1])
        y_pred.append(row[2])
    f1 = f1_score(y_true, y_pred, average=None)
    f1_mean = np.mean(f1)
    print("Micro-averaged F1: {}".format(f1_mean))
    print("Individual scores: {}".format(f1))
    print("Confusion matrix:")
    print(ConfusionMatrix(y_true, y_pred))
