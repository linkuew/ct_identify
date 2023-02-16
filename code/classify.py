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

    seed, seed_eval, mode, feat, n1, n2 = process_args(sys.argv[0])

    print (seed)
    print (seed_eval)

    # check to make sure the LOCO partition exists
    # if not os.path.exists('../data/LOCO_partition.json'):
    #     partition_dataset()

    # open the partitioned data
    # with open('../data/LOCO_partition.json') as f:
    #     data = json.load(f)

    # get the data (i realize this is inefficient)
    # ct = split_data(data)
    # ct = ct[data_index]

    # split the data into x and y vectors
    # xdata, ydata = x_y_split(ct)

    # # split the data into train and test
    # xtrain, xtest, ytrain, ytest = train_test_split(xdata, ydata,
    #                                                 train_size = train_perc,
    #                                                 test_size = test_perc)
    
    if mode == 'merge':
        # test on seed, train on a list without seed
        xtrain, ytrain, xtest,  ytest = read_data_merge(seed,seed_eval)
    else:
        xtrain, ytrain, xtest, ytest  = read_data(seed,seed_eval)

    # vectorize the data
    vectorizer = CountVectorizer(analyzer = feat, ngram_range = (n1, n2))

    # convert data to vectorized form
    vec_xtrain = vectorizer.fit_transform(xtrain)
    vec_xtest = vectorizer.transform(xtest)

    # create SVM model and grid search
    svc = LinearSVC()
    param_grid = {'loss': ['squared_hinge'],
                      'random_state': [1291],
                      'C': [0.01,0.1,1]}

    model = GridSearchCV(svc, param_grid, n_jobs=16, verbose=1, cv=5)            

    # print out our model parameters
    # print(model.get_params)

    # fit data
    model.fit(vec_xtrain, ytrain)

    # predict with our svm model
    preds = model.predict(vec_xtest)

    # check how our model did
    print("SVM accuracy: ", accuracy_score(preds, ytest) * 100)
    print (classification_report(ytest, preds, digits=4))

    out_res = './res_tr_'+seed+'_te_'+seed_eval+'.csv'

    report = classification_report(ytest, preds, digits=4, output_dict= True)
    print (report)
    df = pd.DataFrame(report).transpose()

    df.to_csv(out_res)

if __name__ == "__main__":
    main()
