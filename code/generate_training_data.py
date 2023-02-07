import pandas as pd
import json
from sklearn.model_selection import train_test_split


# get text and label
def x_y_split(data):
    X = []
    Y = []
    df =pd.DataFrame()
    for entry in data:
        X.append(entry['txt'])
        Y.append(entry['subcorpus'])
        
    df['text'] = X
    df['label'] = Y
    return df

# save to train and test files
def get_training_data(data, seed):
    
    docs = []
    for entry in data:
        if entry['seeds'].__contains__(seed):
            docs.append(entry)
    df = x_y_split(docs)
    train, test = train_test_split(df, test_size = 0.3)
    train.to_pickle('../data/train_'+seed+'.pkl')
    test.to_pickle('../data/test_'+seed+'.pkl')


data = json.load(open('../data/LOCO_partition.json'))
seeds = ['big.foot','vaccine','flat.earth','pizzagate','climate']
for seed in seeds:
    get_training_data(data,seed)