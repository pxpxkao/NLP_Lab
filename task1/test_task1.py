import sys
args = sys.argv
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support

'''
Usage:
    python3 test_task1.py [tgt_file] [pred_file]
Example:
    python3 test_task1.py dev.csv fincausal.tsv
    python3 test_task1.py ../data/dev_test/dev.csv xlnet/pred_dir/fincausal.tsv

CSV columns -> Index ; Arg1 ; Arg2 ; **Label**
'''

dev_obj = pd.read_csv(args[1], ';')
dev = np.array(dev_obj.values)[:, 3]
print(dev.shape)
dev = list(dev)

pred_obj = pd.read_csv(args[2], '\t')
pred = np.array(pred_obj.values)[:, 1]
print(pred.shape)
pred = list(pred)

target_names = ['class 0', 'class 1']
print(classification_report(dev, pred, target_names=target_names))

def evaluate(y_true, y_pred):
    """
    Evaluate Precision, Recall, F1 scores between y_true and y_pred
    If output_file is provided, scores are saved in this file otherwise printed to std output.
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: list of scores (F1, Recall, Precision, ExactMatch)
    """
    
    assert len(y_true) == len(y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], average='weighted')
    scores = [
        "F1: %f\n" % f1,
        "Recall: %f\n" % recall,
        "Precision: %f\n" % precision,
        "ExactMatch: %f\n" % -1.0
    ]
    for s in scores:
        print(s, end='')

evaluate(dev, pred)