import pandas as pd
import numpy as np
from collections import defaultdict
from nltk.tokenize import word_tokenize
import nltk
from funcy import lflatten
import re
from sklearn.model_selection import train_test_split

# nltk.word_tokenize : max_len 176
# 3 Labels : "_" ""C" "E"
# 5 Labels : "_" "B-C" "I-C" "B-E" "I-E"

def s2dict(lines, lot):

    """
    :param lines: list of sentences or words as strings containing at least two nodes to be mapped in dict
    :param lot: list of tags to be mapped in the dictionary as keys
    :return: dict with keys == tag and values == sentences /words
    """
    d = defaultdict(list)
    for line_, tag_ in zip(lines, lot):
        d[tag_] = line_

    return d


def make_causal_input(lod, map_, silent=True, bio=False):

    """
    :param lod: list of dictionaries
    :param map_: mapping of tags and values of interest, i.e. [('cause', 'C'), ('effect', 'E')]. The silent tags are by default taggerd as '_'
    :return: dict of list of tuples for each sentence
    """

    dd = defaultdict(list)
    dd_ = []
    rx = re.compile(r"(\b[-']\b)|[\W_]")
    rxlist = [r'("\\)', r'(\\")']
    rx = re.compile('|'.join(rxlist))
    for i in range(len(lod)):
        line_ = lod[i]['sentence']
        line = re.sub(rx, '', line_)
        ante = lod[i]['cause']
        ante = re.sub(rx, '', ante)
        cons = lod[i]['effect']
        cons = re.sub(rx, '', cons)

        silent or print(line)
        d = defaultdict(list)
        index = 0
        for idx, w in enumerate(word_tokenize(line)):
            index = line.find(w, index)

            if not index == -1:
                d[idx].append([w, index])
                silent or print(w, index)

                index += len(w)

        d_= defaultdict(list)
        for idx in d:

            d_[idx].append([tuple([d[idx][0][0], '_']), d[idx][0][1]])

            def cut_space(init_t):
                for s_idx, s in enumerate(line[init_t:]):
                    if s != ' ':
                        init_t += s_idx
                        return init_t

            init_a = cut_space(line.find(ante))
            init_c = cut_space(line.find(cons))
            
            ante_list = word_tokenize(ante)
            for (el_idx, el) in enumerate(ante_list):
                start = line.find(el, init_a)
                # print('start A')
                # print(start)
                # print(int(d_[idx][0][1]))
                stop = line.find(el, init_a) + len(el)
                word = line[start:stop]
                # print(word)
                if int(start) == int(d_[idx][0][1]):
                    und_ = defaultdict(list)
                    if bio and el_idx == 0:
                        und_[idx].append([tuple([word, 'B-C']), line.find(word, init_a)])
                    elif bio:
                        und_[idx].append([tuple([word, 'I-C']), line.find(word, init_a)])
                    else:
                        und_[idx].append([tuple([word, 'C']), line.find(word, init_a)])
                    d_[idx] = und_[idx]
                    break
                # init_a += len(word) # wrong
                init_a = cut_space(init_a+len(word))

            cons_list = word_tokenize(cons)
            for (el_idx, el) in enumerate(cons_list):
                start = line.find(el, init_c)
                # print('start C')
                # print(start)
                # print(int(d_[idx][0][1]))
                stop = line.find(el, init_c) + len(el)
                word = line[start:stop]
                # print(word)
                if int(start) == int(d_[idx][0][1]):
                    und_ = defaultdict(list)
                    if bio and el_idx == 0:
                        und_[idx].append([tuple([word, 'B-E']), line.find(word, init_c)])
                    elif bio:
                        und_[idx].append([tuple([word, 'I-E']), line.find(word, init_c)])
                    else:
                        und_[idx].append([tuple([word, 'E']), line.find(word, init_c)])
                    d_[idx] = und_[idx]
                    break
                # init_c += len(word) # wrong
                init_c = cut_space(init_c+len(word))

        dd[i].append(d_)

    for dict_ in dd:
        dd_.append([item[0][0] for sub in [[j for j in i.values()] for i in lflatten(dd[dict_])] for item in sub])

    return dd_

def get_tokens(doc):
    """
    :param doc:
    :return:
    """
    return [token for (token, label) in doc]

def get_multi_labels(doc):
    """
    :param doc:
    :return:
    """
    return [label for (token, label) in doc]

if __name__ == '__main__':
    # -------------------------------------------------------------------------------- #
    #                                   Make data                                      #
    # ---------------------------------------------------------------------------------#
    df = pd.read_csv('data/PRACTICE/fnp2020-fincausal2-task2.csv', delimiter=';')

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

    def write_file(filename, data):
        with open(filename, 'w') as f:
            for seq in data:
                if len(seq) != len(' '.join(seq).split()):
                    print(seq)
                assert len(seq) == len(' '.join(seq).split())
                f.write(' '.join(seq))
                f.write('\n')
    write_file('./data/task2.train.src', X_train)
    write_file('./data/task2.train.tgt', y_train)
    write_file('./data/task2.val.src', X_test)
    write_file('./data/task2.val.tgt', y_test)
    print("Done!")
