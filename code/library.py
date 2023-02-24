import json
import getopt,sys
import scipy.sparse
import scipy.io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_regression, f_classif
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV

# glob_dict = {'bf' : 0, 'fe' : 1, 'cc' : 2, 'va' : 3, 'pg' : 4}

# if we use the fixed data and read them from seed, please comment out the following line
glob_dict = {'bf' : 'big.foot', 'fe' : 'flat.earth', 'cc' : 'climate', 'va' : 'vaccine', 'pg' : 'pizzagate'}
glob_dict_rev = {value:key for key, value in glob_dict.items()}
ct_lsts = list(glob_dict.keys())
#
# Read data from conspiracy seed 
##
def read_data(seed,eval):
    # tr = pd.read_pickle('../data/data_1/train_'+seed+'.pkl')
    # te = pd.read_pickle('../data/test_'+eval+'.pkl')
    tr = pd.read_csv('../data/data_1/train_'+seed+'.csv')
    te = pd.read_csv('../data/test_'+eval+'.csv')    
    return tr['text'],tr['label'],te['text'],te['label']

def read_data_merge(seed,eval):
    # tr = pd.read_pickle('../data/data_4/train_'+seed+'.pkl')
    # te = pd.read_pickle('../data/test_'+eval+'.pkl')
    tr = pd.read_csv('../data/data_4/train_'+seed+'.csv')
    te = pd.read_csv('../data/test_'+eval+'.csv')
    # print (tr.shape)
    # print (te.shape)   
    return tr['text'],tr['label'],te['text'],te['label']
##
# Prints out the usage statements
##
def usage(script_name):
    print("Usage: python " + script_name + " -d [bf|fe|cl|va|pg] -e [bf|fe|cl|va|pg] -m [merge] -f [char|word] -r i,j",
            end='')
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
    print("-t, train percent, test percent, e.g.: 70,30")
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
        if script_name[0] == 'c':
            optlist, _ = getopt.getopt(sys.argv[1:], "hd:e:m:f:r:")
        elif script_name[0] == 'f':
            optlist, _ = getopt.getopt(sys.argv[1:], "hd:e:m:f:r:k:s:")
        elif script_name[0] == 'b':
            optlist, _ = getopt.getopt(sys.argv[1:], "hd:e:m:p:l:b:")        

        for arg, val in optlist:
            if arg == "-h":
                usage()
            elif arg == "-d":
                dataset = glob_dict.get(val)
            elif arg == "-e":
                eval = glob_dict.get(val)                
            # elif arg == "-t":
            #     tmp = val.split(",")
            #     train = int(tmp[0])
            #     test = int(tmp[1])
            elif arg == "-m":
                mode = val
                if mode == "merge":
                    eval = dataset 
                    ct_lsts.remove(glob_dict_rev[dataset])
                    dataset = '_'.join(ct_lsts)               
            elif arg == "-r":
                tmp = val.split(",")
                low = int(tmp[0])
                upp = int(tmp[1])
            elif arg == "-f":
                feat = val
            elif arg == "-k":
                num_feat = int(val)
            elif arg == "-s":
                func = val
            # bert args, epoc, learning rate, batch
            elif arg == "-p":
                p = int(val)
            elif arg == "-l":
                l = float(val)
            elif arg == "-b":
                batch = int(val)           


    except Exception as e:
        print(e)
        usage()
        exit(-1)

    if script_name[0] == 'c':
        return dataset, eval, mode, feat, low, upp
    elif script_name[0] == 'b':
        return dataset, eval, mode, p, l, batch
    else:
        return dataset, eval, mode, feat, low, upp, num_feat, func


##
# function to partition dataset according to our pre-determined
#
# input: none - just make sure the LOCO.json file is the correct directory
#
# output: none - there should be a new file which is just the conspiracy theories we want
##
def partition_dataset():
    f = open('../data/LOCO.json')
    data = json.load(f)

    new_data = []

    for i in range(len(data)):
        if data[i]['seeds'].__contains__('big.foot') \
            or data[i]['seeds'].__contains__('flat.earth') \
            or data[i]['seeds'].__contains__('climate') \
            or data[i]['seeds'].__contains__('vaccine') \
            or data[i]['seeds'].__contains__('pizzagate'):
            new_data.append(data[i])

    with open('../data/LOCO_partition.json', 'w') as of:
        of.write('[')
        for i in range(len(new_data)):
            tmp_json = json.dumps(new_data[i], indent = 4)
            if i == (len(new_data) - 1):
                of.write(tmp_json)
            else:
                of.write(tmp_json + ",\n")
        of.write(']')

##
# Split the data into the different areas which we want to test on
#
# input: entire corpus
#
# returns: subcorpora
##
def split_data(data):
    vaccine = []
    bigfoot = []
    flat = []
    pizza = []
    climate = []
    for entry in data:
        if entry['seeds'].__contains__('big.foot'):
            bigfoot.append(entry)
        if entry['seeds'].__contains__('vaccine'):
            vaccine.append(entry)
        if entry['seeds'].__contains__('flat.earth'):
            flat.append(entry)
        if entry['seeds'].__contains__('pizzagate'):
            pizza.append(entry)
        if entry['seeds'].__contains__('climate'):
            climate.append(entry)

    return [bigfoot, climate, flat, pizza, vaccine]

##
# A helper function for the conspiracy select, this returns the text entry from the corpus
#
# input: a corpus
#
# output: a breakdown of the input corpus into text (X) and label (Y) - note that 1 marks conspiracy and 0 marks non-conspiracy
##
def x_y_split(data):
    X = []
    Y = []

    for entry in data:
        if entry['subcorpus'] == 'conspiracy':
            X.append(entry['txt'])
            Y.append(1)
        else:
            X.append(entry['txt'])
            Y.append(0)

    return X, Y

##
# Feature selection algorithm
#
# input: filepath to the
#
#
##
def feature_selection(xtrain, ytrain,  xtest,  ytest, kfeature, n1, n2, func, feat, seed, eval_seed, min=10, bi=False):
    # build the filename here
    filename = str(kfeature) + '_' \
            + func + '_' \
            + str(feat) + '_' \
            + str(n1) + '-' + str(n2) + '_' \
            + 'tr_' + seed
    
   # set up vectorizing object
    vectorizer_word = CountVectorizer(analyzer = feat,
            ngram_range=(n1, n2),
            min_df=min,
            binary=bi,
            token_pattern=r'\b\w+\b')

    # fit this vectorizer to the training matrix
    train_bool_matrix = vectorizer_word.fit_transform(xtrain)
    test_bool_matrix = vectorizer_word.transform(xtest)

    # get the features from the vectorizer
    features = vectorizer_word.get_feature_names_out()

    # find the best features according to our function
    bestfeatures = SelectKBest(globals()[func], k = kfeature)

    # update the vectorizer to the best features
    train_new = bestfeatures.fit_transform(train_bool_matrix, ytrain)
    test_new = bestfeatures.transform(test_bool_matrix)

    print(train_new.shape)
    print(test_new.shape)

    fit = bestfeatures.fit(train_bool_matrix,ytrain)
    dfscores = pd.DataFrame(fit.scores_)

    #scipy.io.mmwrite("../results/train_bleach_sel"+str(kfeature)+".mtx", train_new)
    #scipy.io.mmwrite("../results/test_bleach_sel"+str(kfeature)+".mtx", test_new)

    dfcolumns = pd.DataFrame(features)

    #concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns

    # write to results folder
    rank = featureScores.nlargest(1000,'Score')
    rank.to_csv('./' + filename+'.txt')

    # create SVM model and grid search
    svc = LinearSVC()
    param_grid = {'loss': ['squared_hinge'],
                      'random_state': [1291],
                      'C': [0.01,0.1,1]}

    model = GridSearchCV(svc, param_grid, n_jobs=16, verbose=1, cv=5)            

    # print out our model parameters
    # print(model.get_params)

    # fit data
    model.fit(train_new, ytrain)

    # predict with our svm model
    preds = model.predict(test_new)

    # check how our model did
    # dataset eval
    out_res = filename+ '_te_'+eval_seed+'.res.csv'

    report = classification_report(ytest, preds, digits=4, output_dict= True)
    print (report)
    df = pd.DataFrame(report).transpose()

    df.to_csv(out_res)
