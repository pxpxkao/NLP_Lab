import sys
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support
labels = {"C": 1, "E": 2, "_": 0}

def read_file(filename):
    with open(filename, encoding='utf-8') as f:
        data = []
        for line in f.readlines():
            data.append(line.strip().split())
    print('-----------------------------------------------')
    print(filename, ':', len(data))
    return data
        
def write_file(output, data):
    with open(output, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(' '.join(line))
            f.write('\n')

def evaluate(y_pred, y_test):
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

def post_process(pred):
    post_pred = []
    for line in pred:
        post = []
        flag = {'E':0, 'C':0, '_':0}
        for idx, e in enumerate(line):
            if idx == 0:
                post.append(e)
            elif idx == 1 or idx == len(line)-2 or idx == len(line)-1:
                post.append(post[-1])
            else:
                cnt = {'E':0, 'C':0, '_':0}
                for i in range(5):
                    cnt[line[idx-2+i]] += 1
                cnt = sorted(cnt.items(), key = lambda x:x[1], reverse=True)
                if cnt[0][0] == e and cnt[0][1] != cnt[1][1]:
                    post.append(e)
                else:
                    if e != post[-1] and cnt[0][1] == cnt[1][1] and not flag[e] and e != '_':
                        post.append(e)
                    elif e != post[-1] and cnt[0][1] >= 3 and cnt[0][1] == post[-1]:
                        post.append(post[-1])
                    else:
                        post.append(post[-1])
            flag[post[-1]] += 1
        post_pred.append(post)
    write_file('test.txt', post_pred)
    for idx, line in enumerate(post_pred):
        c_pos = get_longest(line, 'C')
        e_pos = get_longest(line, 'E')
        post = np.empty(len(line), dtype=str)
        if c_pos:
            post[c_pos[0]: c_pos[1]+1] = 'C'
        if e_pos:
            post[e_pos[0]: e_pos[1]+1] = 'E'
        post[post == ''] = '_'
        post_pred[idx] = list(post)
    write_file('test1.txt', post_pred)
    return post_pred

def is_equal_len(pred, ref):
    assert len(pred) == len(ref)
    for i in range(len(ref)):
        if len(pred[i]) != len(ref[i]):
            print(i, len(pred[i]), len(ref[i]))
        assert len(pred[i]) == len(ref[i])
    return True

if __name__ == '__main__':
    filename = sys.argv[1]
    pred_file = filename # 'pred_dir/pred_200.txt'
    ref_file = '../data/tags/task2.test.tgt'
    pred = read_file(pred_file)
    ref = read_file(ref_file)
    pred = post_process(pred)
    print(is_equal_len(pred, ref))
    evaluate(pred, ref)
    