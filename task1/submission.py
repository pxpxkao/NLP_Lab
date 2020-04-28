import sys
args = sys.argv
import pandas as pd
import numpy as np
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