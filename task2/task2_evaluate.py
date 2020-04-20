import sys
args = sys.argv
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

'''
Usage:
    python3 test_task2.py data/task2.val.tgt transformer/pred_dir/pred.txt
Input file format:
    task2.val.tgt: E E E E E _ C C C C
    pred.txt: E E E E _ _ C C C C
'''



labels = {"C": 1, "E": 2, "_": 0}

y_test = []
with open(args[1], 'r') as f:
    lines = f.readlines()
    for line in lines:
        y_test.append(line.strip().split())

y_pred = []
with open(args[2], 'r') as f:
    lines = f.readlines()
    for line in lines:
        y_pred.append(line.strip().split())

truths = np.array([labels[tag] for row in y_test for tag in row])
predictions = np.array([labels[tag] for row in y_pred for tag in row])  
print(np.sum(truths == predictions) / len(truths))

# # Print out the classification report
print('************************ classification report ***************************', '\t')
print(classification_report(
    truths, predictions,
    target_names=["_", "C", "E"]))

# # Print out task2 metrics
print('************************ tasks metrics ***************************', '\t')
F1metrics = precision_recall_fscore_support(truths, predictions, average='weighted')
print('F1score:', F1metrics[2])
print('Precision: ', F1metrics[1])
print('Recall: ', F1metrics[0])
cnt = 0
for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        cnt += 1
print('Exact matches: ', cnt, 'over', len(y_pred), 'total sentences...')