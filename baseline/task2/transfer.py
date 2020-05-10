import os
data_dir = 'data'
import pandas as pd
import numpy as np
from make_causal import *
from sklearn.model_selection import train_test_split

def transfer(filename, mode):
    print(filename)
    ################## Make transformer format data #######################
    df = pd.read_csv(os.path.join(data_dir, filename), delimiter=';')
    lodict_, multi_ = [], {}
    for idx, rows in enumerate(df.itertuples()):
        if rows[1].count('.') == 2:
            root_idx = '.'.join(rows[1].split('.')[:-1])
            if root_idx in multi_:
                multi_[root_idx].append(idx)
            else:
                multi_[root_idx] = [idx]
        list_ = [rows[2], rows[3], rows[4]]
        map1 = ['sentence', 'cause', 'effect']
        dict_ = s2dict(list_, map1)
        lodict_.append(dict_)

    if mode == 'train':
        print('transformation example: ', lodict_[3])

    map_ = [('cause', 'C'), ('effect', 'E')]
    hometags = make_causal_input(lodict_, map_)

    data = []
    for i, j in enumerate(hometags):
        data.append([(w, label) for (w, label) in j])

    X = np.array([get_tokens(doc) for doc in data])
    y = np.array([get_multi_labels(doc) for doc in data])
    # for (key, idxs) in multi_.items():
    #     print(key, idxs)
    #     for i, idx in enumerate(idxs):
    #         X[idx] = [str(i)] + X[idx]
    #         y[idx] = ['_'] + y[idx]
    '''
    mask = np.ones(len(X), dtype=bool)
    for (key, idx) in multi_.items():
        print(key, idx)
        text = X[idx[0]]
        tag = y[idx[0]]
        print(''.join(tag))
        for i in range(1, len(idx)):
            assert text == X[idx[i]]
            for t in range(len(tag)):
                if tag[t] != y[idx[i]][t]:
                    if tag[t] == 'C' or y[idx[i]][t] == 'C':
                        tag[t] = 'C'
                    elif tag[t] == 'E' or y[idx[i]][t] == 'E':
                        tag[t] = 'E'
            mask[i] = False
            print(''.join(y[idx[i]]))
            print(''.join(tag))

    X = X[mask]
    y = y[mask]
    '''
    if mode == 'train':
        # split into train, dev
        size = 0.2
        seed = 42
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=seed)

        write_file(os.path.join(data_dir, 'task2.train.src'), X_train)
        write_file(os.path.join(data_dir, 'task2.train.tgt'), y_train)
        write_file(os.path.join(data_dir, 'task2.val.src'), X_test)
        write_file(os.path.join(data_dir, 'task2.val.tgt'), y_test)
        print("Done!")

    elif mode == 'test':
        write_file(os.path.join(data_dir, 'task2.test.src'), X)
        write_file(os.path.join(data_dir, 'task2.test.tgt'), y)
        print("Done!")
    else:
        print('No such mode!')

if __name__ == '__main__':
    transfer('train.csv', 'train')
    transfer('test_gold.csv', 'test')