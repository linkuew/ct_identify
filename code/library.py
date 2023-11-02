import json
import os
import getopt,sys
import scipy.sparse
import scipy.io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_regression, f_classif
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# glob_dict = {'bf' : 0, 'fe' : 1, 'cc' : 2, 'va' : 3, 'pg' : 4}

# if we use the fixed data and read them from seed, please comment out the following line
glob_dict = {'bf' : 'big.foot', 'fe' : 'flat.earth', 'cc' : 'climate', 'va' : 'vaccine', 'pg' : 'pizzagate'}
glob_dict_rev = {value:key for key, value in glob_dict.items()}
ct_lsts = list(glob_dict.keys())
#
# Read data from conspiracy seed
##
def read_data(seed,eval):
    tr = pd.read_csv('../data/train_'+seed+'.csv')
    val = pd.read_csv('../data/val_'+seed+'.csv')
    te = pd.read_csv('../data/test_'+eval+'.csv')

    return tr, te, val

def read_data_merge(seed,eval):
    tr = pd.read_csv('../data/data_4/train_'+seed+'.csv')
    val = pd.read_csv('../data/data_4/val_'+seed+'.csv')
    te = pd.read_csv('../data/test_'+eval+'.csv')

    return tr, te, val


##
# Prints out the usage statements
##
def usage(script_name):
    print("Usage: python " + script_name + " -d [bf|fe|cl|va|pg] -e [bf|fe|cl|va|pg] -m [merge] -f [char|word] -r i,j", end="")
    if script_name[0] == 'f':
        print('-k [num_features]')
        print('-s [selection_function]')
    print()
    print("-d, dataset to use")
    print("\t bf = bigfoot")
    print("\t fe = flat earth")
    print("\t cc = climate change")
    print("\t va = vaccines")
    print("\t pg = pizzagate")
    print("-f, feature set, either 'word' or 'char'")
    # print("-t, test or dev mode, if it's true, then do the prediction")
    print("-r, n-gram range for features, e.g. 1,3")
    if script_name[0] == 'f':
        print("-k, number of top features, choose the target number of features here, e.g. 1000")
        print("-s, selection function for features, either 'chi2' or 'mutual_info_classif'")
    print("-h, print this help page")

    if script_name[0] == 'b':
        print("-ep, epoch")
        print("-lr, learning rate")
    print("-h, print this help page")

    return 0

def process_args(script_name):
    args = []

    try:
        # classification
        if script_name[0] == 'c':
            optlist, _ = getopt.getopt(sys.argv[1:], "hd:e:t:m:f:r:y:v:o:")
        # feautre selectio
        elif script_name[0] == 'f':
            optlist, _ = getopt.getopt(sys.argv[1:], "hd:e:t:m:f:r:k:s:y:o:")
        # bert
        elif script_name[0] == 'b':
                optlist, _ = getopt.getopt(sys.argv[1:], "hd:e:m:p:l:b:o:f:")

        for arg, val in optlist:
            if arg == "-h":
                usage(script_name)
            elif arg == "-d":
                dataset = glob_dict.get(val)
            elif arg == "-e":
                eval = glob_dict.get(val)
            # elif arg == "-t":
            #     pred = val
            elif arg == "-m":
                mode = val
                if mode == "merge":
                    ct_lsts.remove(glob_dict_rev[eval])
                    dataset = '_'.join(ct_lsts)
            elif arg == "-r":
                tmp = val.split(",")
                low = int(tmp[0])
                upp = int(tmp[1])
            elif arg == "-f":
                if val == 'char':
                    feat = val                  
                else:  
                    eval = eval+'.'+val 
                    dataset = dataset+'.'+val 
                    feat = 'word'
            elif arg == "-k":
                num_feat = int(val)
            elif arg == "-s":
                func = val
            elif arg == "-o":
                outpath = val

            # bert args, epoc, learning rate, batch
            elif arg == "-p":
                p = int(val)
            elif arg == "-l":
                l = float(val)
            elif arg == "-b":
                batch = int(val)


    except Exception as e:
        print(e)
        usage(script_name)
        exit(-1)

    if script_name[0] == 'c':
        return dataset, eval, mode, feat, low, upp, outpath
    elif script_name[0] == 'b':
        return dataset, eval, mode, p, l, batch, outpath
    else:
        return dataset, eval, mode, feat, low, upp, num_feat, func, outpath



def feature_selection(train, test, val, kfeature, n1, n2, func, feat, seed, eval_seed, outpath, min=1, bi=False):


   # set up vectorizing object
    vectorizer_word = CountVectorizer(analyzer = feat,
            ngram_range=(n1, n2),
            min_df=min,
            binary=bi,
            token_pattern=r'\b\w+\b')

    # fit this vectorizer to the training matrix
    train_bool_matrix = vectorizer_word.fit_transform(train['text'])
    test_bool_matrix = vectorizer_word.transform(test['text'])

    print(train_bool_matrix.shape)

    # get the features from the vectorizer
    features = vectorizer_word.get_feature_names_out()

    # find the best features according to our function
    bestfeatures = SelectKBest(globals()[func], k = kfeature)

    # update the vectorizer to the best features
    train_new = bestfeatures.fit_transform(train_bool_matrix, train['label'])
    test_new = bestfeatures.transform(test_bool_matrix)



    fit = bestfeatures.fit(train_bool_matrix,train['label'])
    dfscores = pd.DataFrame(fit.scores_)


    dfcolumns = pd.DataFrame(features)

    #concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns

    # write to results folder
    rank = featureScores.nlargest(1000,'Score')

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # build the filename here
    filename = outpath+str(kfeature) + '_' \
            + func + '_' \
            + str(feat) + '_' \
            + str(n1) + '-' + str(n2) + '_' \
            + 'tr_' + seed

    rank.to_csv(filename+'.txt')


    # create a new set combining tr and dev
    train_all = pd.concat([train, val], ignore_index=True)
    split_index = [-1]*train.shape[0]+[0]*val.shape[0]
    pds = PredefinedSplit(test_fold = split_index)


    train_all_matrix = vectorizer_word.transform(train_all['text'])
    train_all_new = bestfeatures.transform(train_all_matrix)




    # create SVM model and grid search
    svc = LinearSVC()
    param_grid = {'loss': ['squared_hinge', 'hinge'],
                      'random_state': [1291, 42],
                      'C': [0.01,0.1,1]}

    model = GridSearchCV(svc, param_grid, n_jobs=16, verbose=1, cv=pds)




    # fit data
    model.fit(train_all_new, train_all['label'])

    # print out our model parameters
    print(model.best_params_)

    # predict with our svm model
    preds = model.predict(test_new)

    # dataset eval


    out_res = outpath+'res_tr_'+seed+'_te_'+eval_seed+str(kfeature)+'.csv'

    report = classification_report(test['label'], preds, digits=4, output_dict= True)

    df = pd.DataFrame(report).transpose()

    df.to_csv(out_res)
