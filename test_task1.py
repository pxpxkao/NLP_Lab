import sys
args = sys.argv
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

'''
Usage:
    python3 test_task1.py [tgt_file] [pred_file]
Example:
    python3 test_task1.py dev.csv fincausal.tsv

CSV columns -> Index ; Arg1 ; Arg2 ; **Label**
'''

y_true = [0, 1, 2, 2, 0]
y_pred = [0, 0, 2, 2, 0]
target_names = ['class 0', 'class 1', 'class 2']
# print(classification_report(y_true, y_pred, target_names=target_names))

dev_obj = pd.read_csv(args[1], ';')
dev = np.array(dev_obj.values)[:, 3]
print(dev.shape)
print(dev[0])
dev = list(dev)

pred_obj = pd.read_csv(args[2], '\t')
pred = np.array(pred_obj.values)[:, 1]
print(pred.shape)
print(pred[0])
pred = list(pred)

target_names = ['class 0', 'class 1']
print(classification_report(dev, pred, target_names=target_names))