import sys
args = sys.argv
import pandas as pd
import numpy as np
from sklearn import metrics
'''
Usage:
    python3 submission.py [predictions file] [output file]
    ex. python3 submission.py pred.txt result.csv
'''

submission = pd.read_csv('sample_submission.csv', dtype={'Index':np.dtype(str)})

with open(args[1], 'r', encoding='utf-8') as f:
    pred = []
    for p in f.readlines():
        pred.append(int(p.strip()))

print('Length :', len(pred))

submission.Gold = pred

submission.to_csv(args[2], ',', index=0)

def evaluate(y_true, y_pred):
    """
    Evaluate Precision, Recall, F1 scores between y_true and y_pred
    If output_file is provided, scores are saved in this file otherwise printed to std output.
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: list of scores (F1, Recall, Precision, ExactMatch)
    """
    
    assert len(y_true) == len(y_pred)
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], average='weighted')
    scores = [
        "F1: %f\n" % f1,
        "Recall: %f\n" % recall,
        "Precision: %f\n" % precision,
        "ExactMatch: %f\n" % -1.0
    ]
    for s in scores:
        print(s, end='')

true = pd.read_csv('data/test_gold.csv', ';')
true = list(true.Gold)
evaluate(true, pred)