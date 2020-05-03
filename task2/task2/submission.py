import sys
args = sys.argv
import pandas as pd
import numpy as np
import nltk
'''
python3 submission.py text.txt predictions_tags.txt
'''

def readfile(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = []
        for line in f.readlines():
            data.append(line.strip())
    return data

def get_longest(line, tag):
    longest, p = [], [-1, 0]
    for idx, e in enumerate(line):
        if e == tag and p[0] == -1:
            p[0] = idx
        elif e == tag and idx == len(line)-1 and p[0] != -1:
            p[1] = idx
            longest.append(p)
        elif e != tag and p[0] != -1:
            p[1] = idx - 1
            longest.append(p)
            p = [-1, 0]
    longest = sorted(longest, key = lambda x:x[1]-x[0]+1, reverse=True)
    if len(longest):
        return longest[0]
    else:
        return None

def post_process(pred, text, offset_2, offset_3):
    cause, effect = ['' for i in range(len(pred))], ['' for i in range(len(pred))]
    for idx, line in enumerate(pred):
        line = line.split()
        text[idx] = nltk.word_tokenize(text[idx])
        c_pos = get_longest(line, 'C')
        e_pos = get_longest(line, 'E')
        post = np.empty(len(line), dtype=str)
        if c_pos:
            post[c_pos[0]: c_pos[1]+1] = 'C'
            cause[idx] = ' '.join(text[idx][c_pos[0]: c_pos[1]+1])
        if e_pos:
            post[e_pos[0]: e_pos[1]+1] = 'E'
            effect[idx] = ' '.join(text[idx][e_pos[0]: e_pos[1]+1])
        post[post == ''] = '_'
        pred[idx] = list(post)

    return cause, effect

if __name__ == '__main__':
    text = readfile(args[1])
    tag = readfile(args[2])
    n = 2

    sub = pd.read_csv('data/test.csv',';')
    sub = sub.astype({'Offset_Sentence2': pd.Int64Dtype(), 'Offset_Sentence3': pd.Int64Dtype()})

    offset_2, offset_3 = sub.Offset_Sentence2, sub.Offset_Sentence3
    cause, effect = post_process(tag, text, offset_2, offset_3)
    # print('Index:\n\t', sub.Index[n])
    # print('text:\n\t', text[n])
    # print('tag:\n\t', tag[n])
    # print('Cause:\n\t', cause[n])
    # print('Effect:\n\t', effect[n])

    sub = sub.drop(columns=['Offset_Sentence2', 'Offset_Sentence3'])
    sub['Cause'] = cause
    sub['Effect'] = effect
    # print(sub.columns)

    sub.to_csv('task2_pred.csv', ';', index=0)