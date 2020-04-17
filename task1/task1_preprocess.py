import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pandas import DataFrame

def bertIn(): 
	le = LabelEncoder()
	#read source data from csv file
	df_train = pd.read_csv('./data/train.csv', sep = '; ', engine='python')
    df_test = pd.read_csv('./data/test_gold.csv', sep = '; ', engine='python')

	#create a new dataframe for train, dev data
	df_bert = pd.DataFrame({'id': np.arange(len(df_train)),
	'label': le.fit_transform(df_train['Gold']),
	'alpha': ['a']*df_train.shape[0],
	'text': df_train['Text']})

	#split into train, dev
	df_bert_train, df_bert_dev = train_test_split(df_bert, test_size=0.2, random_state=0)

	#create new dataframe for test data
	df_bert_test = pd.DataFrame({'id': np.arange(len(df_test)),
	'text': df_test['Text'],
    'label': df_test['Gold']})

	#output tsv file, no header for train and dev
	df_bert_train.to_csv('./data/train.tsv', sep='\t', index=False, header=False)
	print("Train data loaded")
	df_bert_dev.to_csv('./data/dev.tsv', sep='\t', index=False, header=False)
	print("Dev data loaded")
	df_bert_test.to_csv('./data/test.tsv', sep='\t', index=False, header=False)
	print("Test data loaded")

if __name__ == "__main__":
	bertIn()