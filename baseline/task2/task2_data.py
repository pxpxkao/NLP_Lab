import os
import numpy as np
import pandas as pd # 0.24+
from make_causal import *
from sklearn.model_selection import train_test_split

data_dir = 'data'
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

def randomly_prepare_data():
    df_data = pd.read_csv('data/fnp2020-fincausal2-task2.csv', sep='; ', engine='python')

    df_train, df_test = train_test_split(df_data, test_size=0.2, random_state=0)

    df_train.to_csv(os.path.join(data_dir, 'train.csv'), ';', index=0)
    df_test.to_csv(os.path.join(data_dir, 'test_gold.csv'), ';', index=0)

    print(df_test.columns)
    df_test = df_test.drop(columns=['Cause', 'Effect', 'Cause_Start', 'Cause_End', 'Effect_Start', 'Effect_End', 'Sentence'])
    df_test.to_csv(os.path.join(data_dir, 'test.csv'), ';', index=0)

def prepare_data():
    df_data = pd.read_csv('data/fnp2020-fincausal2-task2.csv', sep='; ', engine='python')
    df_data = df_data.astype({'Offset_Sentence2': pd.Int64Dtype(), 'Offset_Sentence3': pd.Int64Dtype()})
    # df[['Offset_Sentence2', 'Offset_Sentence3']] = df[['Offset_Sentence2', 'Offset_Sentence3']].fillna(-1)
    # df[['Offset_Sentence2', 'Offset_Sentence3']] = df[['Offset_Sentence2', 'Offset_Sentence3']].astype(int)
    # df[['Offset_Sentence2', 'Offset_Sentence3']] = df[['Offset_Sentence2', 'Offset_Sentence3']].astype(str)
    # df[['Offset_Sentence2', 'Offset_Sentence3']] = df[['Offset_Sentence2', 'Offset_Sentence3']].replace('-1', np.nan)
    print(df_data.dtypes)

    train = pd.read_csv('../task1/data/train.csv', ';')
    train = train.sort_values(by=['Index'])
    train = train[train.Gold == 1]

    test = pd.read_csv('../task1/data/test_Gold.csv', ';')
    test = test.sort_values(by=['Index'])
    test = test[test.Gold == 1]

    df_train = pd.DataFrame(columns=df_data.columns)
    for idx, row in df_data.iterrows():
        for i, r in train.iterrows():
            if row.Text == r.Text:
                df_train = df_train.append(df_data.iloc[idx])
                break
    print(df_train.info())
    df_train.to_csv('data/train.csv', ';', index=0)

    df_test = pd.DataFrame(columns=df_data.columns)
    for idx, row in df_data.iterrows():
        for i, r in test.iterrows():
            if row.Text == r.Text:
                df_test = df_test.append(df_data.iloc[idx])
                break
    print(df_test.info())
    df_test.to_csv('data/test_gold.csv', ';', index=0)

    print(df_test.columns)
    df_test = df_test.drop(columns=['Cause', 'Effect', 'Cause_Start', 'Cause_End', 'Effect_Start', 'Effect_End', 'Sentence'])
    df_test.to_csv('data/test.csv', ';', index=0)

if __name__ == '__main__':
    prepare_data()