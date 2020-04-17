import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data_dir = 'data'
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

df_data = pd.read_csv('data/fnp2020-fincausal2-task1.csv', sep = '; ', engine='python')

# split into train, test
df_train, df_test = train_test_split(df_data, test_size=0.2, random_state=0, stratify=df_data.Gold.values)

df_train.to_csv(os.path.join(data_dir, 'train.csv'), sep=';', index=0, header=['Index', 'Text', 'Gold'])
df_test.to_csv(os.path.join(data_dir, 'test_gold.csv'), sep=';', index=0, header=['Index', 'Text', 'Gold'])

# split into train, dev
# df_train_cut, df_dev_cut = train_test_split(df_train, test_size=0.2, random_state=0, stratify=df_train.Gold.values)
# df_train_cut.to_csv(os.path.join(data_dir, 'train_cut.csv'), ';', index=0, header=['Index', 'Text', 'Gold'])
# df_dev_cut.to_csv(os.path.join(data_dir, 'dev_cut.csv'), ';', index=0, header=['Index', 'Text', 'Gold'])

df_test = df_test.drop(columns=['Gold'])
df_test.to_csv(os.path.join(data_dir, 'test.csv'), sep=';', index=0, header=['Index', 'Text'])