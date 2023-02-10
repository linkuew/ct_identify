import pandas as pd
import json
from sklearn.model_selection import train_test_split
import itertools

'''
STAT OF CT

mainstream    2019
conspiracy     708
Name: label, dtype: int64
mainstream    5139
conspiracy    1965
Name: label, dtype: int64
mainstream    1646
conspiracy     605
Name: label, dtype: int64
mainstream    1012
conspiracy     359
Name: label, dtype: int64
mainstream    2166
conspiracy     889
Name: label, dtype: int64

So, we set 350, 1000
'''


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
    new_df = get_samples(df)
    train, test = train_test_split(new_df, test_size = 0.2)
    train.to_pickle('../data/data_1/train_'+seed+'.pkl')
    test.to_pickle('../data/data_1/test_'+seed+'.pkl')


def get_samples(df):
    # 350 ct
    # 1000 non ct
    
    # shuffle df
    df.sample(frac=1).reset_index(drop=True)

    df_ct = df.loc[df.label == 'conspiracy'].iloc[:350,:]
    df_mn = df.loc[df.label == 'mainstream'].iloc[:1000,:]
    new_df = pd.concat([df_ct,df_mn])
    return new_df.sample(frac=1).reset_index(drop=True)

# merge train data from four CTs into one tr set

def merge_tr(cmb_seed_lst):
    df_lst = []
    for s in cmb_seed_lst:
        tr = pd.read_pickle('../data/data_1/train_'+glob_dict[s]+'.pkl')
        # 270, ((1000+356)*0.8)/4
        tr.sample(frac=1).reset_index(drop=True)
        df_lst.append(tr.iloc[:270,:])
    df = pd.concat(df_lst)
    df.to_pickle('../data/data_4/train_'+'_'.join(cmb_seed_lst)+'.pkl')
        

glob_dict = {'bf' : 'big.foot', 'fe' : 'flat.earth', 'cc' : 'climate', 'va' : 'vaccine', 'pg' : 'pizzagate'}
seed_lst = ['bf', 'fe', 'cc', 'va', 'pg']
seeds = ['big.foot','vaccine','flat.earth','pizzagate','climate']

data = json.load(open('../data/LOCO_partition.json'))


all_combs = list(itertools.combinations(seed_lst, 4))

for seed in seeds:
    get_training_data(data,seed)

for i in all_combs:
    merge_tr(list(i))


