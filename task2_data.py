import os
import numpy as np
import pandas as pd
from make_causal import *

if not os.path.exists('task2'):
    os.mkdir('task2')

df = pd.read_csv('data/PRACTICE/fnp2020-fincausal2-task2.csv', ';')

data = []
for rows in df.itertuples():
    data.append(rows[1:])
data = np.array(data)
print(data.shape)
seed = 42
np.random.seed(seed)
np.random.shuffle(data)

X_train = data[len(data)//5:]
X_test = data[:len(data)//5]

df = pd.DataFrame(X_train, columns=['Index', 'Text', 'Cause', 'Effect', 'Offset_Sentence2', 'Offset_Sentence3', 'Cause_Start', 'Cause_End', 'Effect_Start', 'Effect_End', 'Sentence'])
df.to_csv('task2/train.csv', ';', index=0)
df = pd.DataFrame(X_test, columns=['Index', 'Text', 'Cause', 'Effect', 'Offset_Sentence2', 'Offset_Sentence3', 'Cause_Start', 'Cause_End', 'Effect_Start', 'Effect_End', 'Sentence'])
df.to_csv('task2/test_gold.csv', ';', index=0)
print(X_test.shape)

X_test = np.concatenate((np.array(X_test[:, :2]), np.array(X_test[:, 4:6])), axis=1)
df = pd.DataFrame(X_test, columns=['Index', 'Text', 'Offset_Sentence2', 'Offset_Sentence3'])
df.to_csv('task2/test.csv', ';', index=0)
print(X_test.shape)

################## Make transformer format data #######################
df = pd.read_csv('task2/train.csv', delimiter=';')
lodict_ = []
for rows in df.itertuples():
    list_ = [rows[2], rows[3], rows[4]]
    map1 = ['sentence', 'cause', 'effect']
    dict_ = s2dict(list_, map1)
    lodict_.append(dict_)

print('transformation example: ', lodict_[3])

map_ = [('cause', 'C'), ('effect', 'E')]
hometags = make_causal_input(lodict_, map_)

data = []
for i, j in enumerate(hometags):
    data.append([(w, label) for (w, label) in j])

X = [get_tokens(doc) for doc in data]
y = [get_multi_labels(doc) for doc in data]
# print(X[3], len(X[3]))
# print(y[3], len(y[3]))

size = 0.2
seed = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=seed)

write_file('task2/task2.train.src', X_train)
write_file('task2/task2.train.tgt', y_train)
write_file('task2/task2.val.src', X_test)
write_file('task2/task2.val.tgt', y_test)
print("Done!")
