import os
import numpy as np
import pandas as pd

if not os.path.exists('task1'):
    os.mkdir('task1')

df = pd.read_csv('data/PRACTICE/fnp2020-fincausal2-task1.csv', ';')

data = []
for rows in df.itertuples():
    idx = format(rows[1], '.5f')
    while(len(idx) < 10):
        idx = '0' + idx
    data.append([idx, rows[2], rows[3]])
data = np.array(data)
print(data.shape)
seed = 42
np.random.seed(seed)
np.random.shuffle(data)

X_train = data[len(data)//5:]
X_test = data[:len(data)//5]
print(X_train.shape, X_test.shape)
df = pd.DataFrame(X_train, columns=['Index', 'Text', 'Gold'])
df.to_csv('task1/train.csv', ';', index=0)

X_dev = X_train[:len(X_train)//5]
X_train = X_train[len(X_train)//5:]
print(X_dev.shape, X_train.shape)
df = pd.DataFrame(X_train, columns=['Index', 'Text', 'Gold'])
df.to_csv('task1/train_cut.csv', ';', index=0)
df = pd.DataFrame(X_dev, columns=['Index', 'Text', 'Gold'])
df.to_csv('task1/dev_cut.csv', ';', index=0)
df = pd.DataFrame(X_test, columns=['Index', 'Text', 'Gold'])
df.to_csv('task1/test_gold.csv', ';', index=0)

X_test = np.array(X_test)[:, :2]
df = pd.DataFrame(X_test, columns=['Index', 'Text'])
df.to_csv('task1/test.csv', ';', index=0)