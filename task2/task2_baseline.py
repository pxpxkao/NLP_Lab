import argparse
from utils import *
import pandas as pd
from collections import defaultdict, Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
import pycrfsuite
import os
import pickle
import nltk
import csv

# ------------------------------------------------------------------------------------ #
#               This baseline uses nltk tokenizer and POS tagger                       #
# -------------------------------------------------------------------------------------#

print('nltk POS tagger example')
text = "The Sunshine State drew in a net influx of about $17.7 billion in adjusted gross income (AGI)  -  most of which (72 percent) came from those aged 55 and older"
print('************************ sentence *************************', '\t')
print(text)
tokens = nltk.word_tokenize(text)
print('************************ tokens ***************************', '\t')

print(tokens)
tag = nltk.pos_tag(tokens)
print('************************ POS tags *************************', '\t')

print(tag)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--inrepo', type = str, default="./data/train.csv", help= 'input repo')
    parser.add_argument('--predrepo', type = str, default="./data/task2.csv", help= 'test repo')
    parser.add_argument('--verbose', type = bool, default=False, help='print training')

    parser.add_argument('--idx', type = str, default="baseline", help= 'experience index')
    # --------------------------------------------------------------------------------- #
    #             if the arguments idx is used, assumes all following arguments         #
    # ----------------------------------------------------------------------------------#
    parser.add_argument('--c1', type=float, help='coefficient for L1 penalty')
    parser.add_argument('--minfreq', type=float, help='')
    parser.add_argument('--featstate', type=bool, help='')
    parser.add_argument('--c2', type=float, help='coefficient for L2 penalty')
    parser.add_argument('--maxiter', type=str, help='maximum number of iterations')
    parser.add_argument('--trans', type=bool, help='whether to include transitions that are possible, but not observed')

    args = parser.parse_args()

    # --------------------------------------------------------------------------------- #
    #                                     Make data                                     #
    # ----------------------------------------------------------------------------------#
    def read_data(filename, train=True):
        df = pd.read_csv(filename, delimiter=';', header=0)

        lodict_ = []
        for rows in df.itertuples():
            if train:
                list_ = [rows[2], rows[3], rows[4]]
                map1 = ['sentence', 'cause', 'effect']
            else:
                list_ = [rows[2]]
                map1 = ['sentence']
            dict_ = s2dict(list_, map1)
            lodict_.append(dict_)

        print('transformation example: ', lodict_[1])

        if train:
            map_ = [('cause', 'C'), ('effect', 'E')]
            hometags = make_causal_input(lodict_, map_)
        postags = nltkPOS([i['sentence'] for i in lodict_])
        sent = [i['sentence'] for i in lodict_]

        data = []
        if train:
            for i, (j, k) in enumerate(zip(hometags, postags)):
                data.append([(w, pos, label) for (w, label), (word, pos) in zip(j, k)])
        else:
            for j in postags:
                data.append([(w, pos) for (w, pos) in j])
        
        X = [extract_features(doc) for doc in data]
        if train:
            return X
        y = [get_multi_labels(doc) for doc in data]
        return X, y, sent

    # --------------------------------------------------------------------------------- #
    #                              Make train and test sets                             #
    # ----------------------------------------------------------------------------------#
    X_train, y_train, X_train_sent = read_data(args.inrepo)
    X_test = read_data(args.predrepo, False)
    print('Length of Xtrain:', len(X_train))
    print('Length of Xtest:', len(X_test))

    
    '''
    X, y, sent = read_data(args.inrepo)
    size = 0.2
    seed = 42
    n = 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=seed)
    X_train_sent, X_test_sent = train_test_split(sent, test_size=size, random_state=seed)
    '''

    # Declare trainer
    trainer = pycrfsuite.Trainer(verbose=args.verbose)

    # Submit training data to the trainer
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)


    # --------------------------------------------------------------------------------- #
    #                                 Set model parameters                              #
    # ----------------------------------------------------------------------------------#
    # trainer.select("l2sgd")
    if args.idx:

        trainer.set_params({
            'c1': args.c1,
            'feature.minfreq': args.minfreq,
            'feature.possible_states': args.featstate,
            'c2': args.c2,
            'max_iterations': args.maxiter,
            'feature.possible_transitions': args.trans
        })

    else:
        trainer.set_params({
            'c1': 0.1,
            'feature.minfreq': 0.7,
            'feature.possible_states': True,
            'c2': 0.01,
            'max_iterations': "3000",
            'feature.possible_transitions': False
        })


    # --------------------------------------------------------------------------------- #
    #                                       Train                                       #
    # ----------------------------------------------------------------------------------#

    # The model will be saved to ./models when training is finished, with crf_args.idx.model name
    if not os.path.exists('models'):
        os.makedirs("models")
    modelpath_ = os.path.join("models", str(args.idx))
    if not os.path.exists(modelpath_):
        os.makedirs(modelpath_)
    trainer.train(os.path.join(modelpath_, ("crf_" + str(args.idx)) + ".model"))

    # The data will be dumped to ./models when training is finished, with data_args.idx.dat name
    data_list = [X_train, X_test, y_train, y_test]

    if not os.path.exists('data'):
        os.makedirs("data")
    datapath_ = os.path.join("data", str(args.idx))
    if not os.path.exists(datapath_):
        os.makedirs(datapath_)
    pickle.dump(data_list, open(os.path.join(datapath_, ("data_" + str(args.idx)) + ".dat"), "wb"))

    # Declare a tagger to predict tags in new text entries
    tagger = pycrfsuite.Tagger()

    # Load model in tagger
    tagger.open(os.path.join(modelpath_, ("crf_" + str(args.idx)) + ".model"))


    # --------------------------------------------------------------------------------- #
    #                                       Predict                                     #
    # ----------------------------------------------------------------------------------#

    y_pred = [tagger.tag(xseq) for xseq in X_test]
    with open('predictions_tags.txt', 'w', encoding='utf-8') as f:
        for pred in y_pred:
            f.write(' '.join(pred))
            f.write('\n')
    with open('text.txt', 'w', encoding='utf-8') as f:
        for text in X_test_sent:
            f.write(text)
            f.write('\n')
    with open('truth_tags.txt', 'w', encoding='utf-8') as f:
        for truth in y_test:
            f.write(' '.join(truth))
            f.write('\n')
    labels = {"C": 1, "E": 2, "_": 0}

    # Convert the sequences of tags into a 1-dimensional array
    predictions = np.array([labels[tag] for row in y_pred for tag in row])
    truths = np.array([labels[tag] for row in y_test for tag in row])
    print(np.sum(truths == predictions) / len(truths))

'''
    # --------------------------------------------------------------------------------- #
    #                                  Print metrics                                    #
    # ----------------------------------------------------------------------------------#

    # # Print out the classification report
    print('************************ classification report ***************************', '\t')
    print(classification_report(
        truths, predictions,
        target_names=["_", "C", "E"]))

    # # Print out task2 metrics
    print('************************ tasks metrics ***************************', '\t')

    F1metrics = precision_recall_fscore_support(truths, predictions, average='weighted')
    # print results and make tagged sentences
    ll = []
    for i in range(len(X_test) - 1):
        l = defaultdict(list)
        for j, (y, x) in enumerate(zip(y_pred[i], list(zip(*[[v for k, v in x.items()] for x in X_test[i]]))[1])):
            l.update({x: y})
        ll.append(l)

    nl = []
    for line, yt, yp, s in zip(ll, y_test, y_pred, X_test_sent):
        d_ = defaultdict(list)
        d_["origin"] = s
        d_["truth"] = yt
        d_["pred"] = yp
        d_["diverge"] = 0
        for k, v in line.items():
            d_[v].append(k)
        if d_["truth"] != d_["pred"]:
            d_["diverge"] = 1
        nl.append(d_)

    print('F1score:', F1metrics[2])
    print('Precision: ', F1metrics[1])
    print('Recall: ', F1metrics[0])
    print('exact match: ', len(nl) - sum([i["diverge"] for i in nl if i['diverge']==1]), 'over', len(nl), ' total sentences)')

    fieldn = sorted(list(set(k for d in nl for k in d)))
    with open(os.path.join(modelpath_, ("predictions_" + str(args.idx)) + ".csv"), "w+", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldn, delimiter="~")
        writer.writeheader()
        for line in nl:
            writer.writerow(line)

    # # Print out other metrics
    print('************************ crf metrics ***************************', '\t')

    trainer.logparser.last_iteration
    info = tagger.info()

    def print_transitions(trans_features):
        for (label_from, label_to), weight in trans_features:
            print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

    print("Top likely transitions:")
    print_transitions(Counter(info.transitions).most_common(3))

    print("\nTop unlikely transitions:")
    print_transitions(Counter(info.transitions).most_common()[-3:])

    print("all transitions:")
    print_transitions(Counter(info.transitions).most_common())

    def print_state_features(state_features):
        for (attr, label), weight in state_features:
            print("%0.6f %-6s %s" % (weight, label, attr))


    print("Top positive:")
    print_state_features(Counter(info.state_features).most_common(20))

    print("\nTop negative:")
    print_state_features(Counter(info.state_features).most_common()[-20:])
'''
