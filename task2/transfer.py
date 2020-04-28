import os
data_dir = 'data'
import pandas as pd
from make_causal import *
from sklearn.model_selection import train_test_split

################## Make transformer format data #######################
df = pd.read_csv(os.path.join(data_dir, 'train.csv'), delimiter=';')
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

# split into train, dev
size = 0.2
seed = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=seed)

write_file(os.path.join(data_dir, 'task2.train.src'), X_train)
write_file(os.path.join(data_dir, 'task2.train.tgt'), y_train)
write_file(os.path.join(data_dir, 'task2.val.src'), X_test)
write_file(os.path.join(data_dir, 'task2.val.tgt'), y_test)
print("Done!")
