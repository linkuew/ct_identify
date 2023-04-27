import pandas as pd
import json
from sklearn.model_selection import train_test_split
import itertools
import copy
'''
STAT OF CT

big.foot
mainstram: 2009
conspiracy: 658
vaccine
mainstram: 5126
conspiracy: 1902
flat.earth
mainstram: 1625
conspiracy: 552
pizzagate
mainstram: 1004
conspiracy: 332
climate
mainstram: 2158
conspiracy: 836

So, we use the smallest conspiracy to set the size of our dataset
330, 1000; because 1330*0.8 is an integer
'''



# save to train and test files
def get_training_data(data, seed, seeds):

    seed_lst = copy.deepcopy(seeds)
    seed_lst.remove(seed)

    docs = []
    for entry in data:
        item_seed = entry['seeds']
        if item_seed.__contains__(seed):
            # check overlap with others
            inersection = [item for item in seed_lst if item_seed.__contains__(item)]
            if len(inersection) == 0:
                docs.append(entry)
    df = get_elements(docs)

    # check_doc_numbers(df)
    
    new_df = get_samples(df)
    # Define the split ratios 
    train_ratio = 0.8
    validation_ratio = 0.1
    test_ratio = 0.1

    # First, split the data into train and a temporary set (validation + test)
    train, temp = train_test_split(new_df, test_size=(validation_ratio + test_ratio), random_state=42)

    # Then, split the temporary set into validation and test sets
    val, test = train_test_split(temp, test_size=test_ratio/(test_ratio + validation_ratio), random_state=42)

    train.to_csv('../data/train_'+seed+'.csv')
    val.to_csv('../data/val_'+seed+'.csv')
    test.to_csv('../data/test_'+seed+'.csv')


# get text and label, and doc id
def get_elements(data):
    X = []
    Y = []
    Z = []
    S = []

    df =pd.DataFrame()
    for entry in data:
        X.append(entry['txt'])
        Y.append(entry['subcorpus'])
        Z.append(entry['doc_id'])
        S.append(entry['seeds'])

        
    df['text'] = X
    df['label'] = Y
    df['doc_id'] = Z
    df['seeds'] = S
    return df


def check_doc_numbers(df):
    mainstream_counts = df['label'].value_counts()['mainstream']
    conspiracy_counts = df['label'].value_counts()['conspiracy']
    print ('mainstram: '+str(mainstream_counts))
    print ('conspiracy: '+str(conspiracy_counts))

def get_samples(df):
    # 330 ct
    # 1000 non ct
    
    # shuffle df
    df.sample(frac=1).reset_index(drop=True)

    df_ct = df.loc[df.label == 'conspiracy'].iloc[:330,:]
    df_mn = df.loc[df.label == 'mainstream'].iloc[:1000,:]
    new_df = pd.concat([df_ct,df_mn])

    return new_df.sample(frac=1).reset_index(drop=True)

# merge train data from four CTs into one tr set

def merge_tr(cmb_seed_lst):
    df_lst = []
    for s in cmb_seed_lst:
        tr = pd.read_csv('../data/train_'+glob_dict[s]+'.csv')
        # 266, ((1000+330)*0.8)/4
        tr.sample(frac=1).reset_index(drop=True)
        df_lst.append(tr.iloc[:266,:])
    df = pd.concat(df_lst)
    # df.to_pickle('../data/data_4/train_'+'_'.join(cmb_seed_lst)+'.pkl')
    df.to_csv('../data/data_4/train_'+'_'.join(cmb_seed_lst)+'.csv')

def merge_val(cmb_seed_lst):
    df_lst = []
    for s in cmb_seed_lst:
        val = pd.read_csv('../data/val_'+glob_dict[s]+'.csv')
        # 33.25, ((1000+330)*0.1)/4
        val.sample(frac=1).reset_index(drop=True)
        df_lst.append(val.iloc[:33,:])
    df = pd.concat(df_lst)
    # df.to_pickle('../data/data_4/train_'+'_'.join(cmb_seed_lst)+'.pkl')
    df.to_csv('../data/data_4/val_'+'_'.join(cmb_seed_lst)+'.csv')
        

glob_dict = {'bf' : 'big.foot', 'fe' : 'flat.earth', 'cc' : 'climate', 'va' : 'vaccine', 'pg' : 'pizzagate'}
seed_lst = ['bf', 'fe', 'cc', 'va', 'pg']
seeds = ['big.foot','vaccine','flat.earth','pizzagate','climate']

data = json.load(open('../data/LOCO_partition.json'))


all_combs = list(itertools.combinations(seed_lst, 4))

for seed in seeds:
    get_training_data(data,seed,seeds)

for i in all_combs:
    merge_tr(list(i))

for i in all_combs:
    merge_val(list(i))
