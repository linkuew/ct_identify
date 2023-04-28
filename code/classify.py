import os
import json
import sys
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV

from library import *
from feature_selection_template import *

############################################################################
#
# Hey everyone, here's a main file which you can use to run experiments
# without knowing too much coding knowledge.
#
# Look through the README.md file before you try running this!
#
# First run: $ python main.py -h
#
############################################################################


# Feb 6th: commented one line for reading the existing data by seeds, see line 56
# Feb 6th: added gridsearch and classfication report\
# Feb 10th: update training and testing data
# Feb 10th: add two modes: train on 1 test on 4, and train on 4 test on 1
# ex1: python classify.py -d bf -e fe -m merge -f word -r 1,3
# ex2: python classify.py -d bf -e fe -m one -f word -r 1,3

def main():
    if len(sys.argv) < 10:
        print (len(sys.argv))
        usage(sys.argv[0])
        exit(1)

    seed, seed_eval, mode, feat, n1, n2,outpath = process_args(sys.argv[0])

    print (seed)
    print (seed_eval)

    if not os.path.exists(outpath):
        os.makedirs(outpath)    
    
    if mode == 'merge':
        # test on seed, train on a list without seed
        tr, te, val = read_data_merge(seed,seed_eval)
    else:
        tr, te, val  = read_data(seed,seed_eval)

    # vectorize the data
    vectorizer = CountVectorizer(analyzer = feat, ngram_range = (n1, n2), min_df = 1, binary=False,token_pattern=r'\b\w+\b')

    # convert data to vectorized form
    vec_xtrain = vectorizer.fit_transform(tr['text'])
    vec_xtest = vectorizer.transform(te['text'])

    print(vec_xtrain.shape)

    # create a new set combining tr and dev
    train_all = pd.concat([tr, val], ignore_index=True)
    split_index = [-1]*tr.shape[0]+[0]*val.shape[0]
    pds = PredefinedSplit(test_fold = split_index)


    train_all_matrix = vectorizer.transform(train_all['text'])


    # create SVM model and grid search
    svc = LinearSVC()
    param_grid = {'loss': ['squared_hinge', 'hinge'],
                      'random_state': [1291, 42],
                      'C': [0.01,0.1,1]}

    model = GridSearchCV(svc, param_grid, n_jobs=16, verbose=1, cv=pds)           



    # fit data
    model.fit(train_all_matrix, train_all['label'])

    # print out our model parameters
    print(model.best_params_)

    with open(outpath+'res_params.csv','a') as out_param:
        out_param.write(seed+'\n')
        out_param.write(seed_eval+'\n')
        out_param.write(str(model.best_params_)+'\n\n')


    # predict with our svm model
    preds = model.predict(vec_xtest)

    # check how our model did
    print("SVM accuracy: ", accuracy_score(te['label'], preds) * 100)
    print (classification_report(te['label'], preds, digits=4))


    out_res = outpath+'res_tr_'+seed+'_te_'+seed_eval+'.csv'

    report = classification_report(te['label'], preds, digits=4, output_dict= True)
    df = pd.DataFrame(report).transpose()

    df.to_csv(out_res)

if __name__ == "__main__":
    main()
