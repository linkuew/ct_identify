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
# Feb 6th: added gridsearch and classfication report
# Feb 10th: added default to use pickled data


def main():
    if len(sys.argv) != 9:
        usage(sys.argv[0])
        exit(1)

    data_index, train_perc, test_perc, feat, n1, n2 = process_args(sys.argv[0])

#    # check to make sure the LOCO partition exists
#    if not os.path.exists('../data/LOCO_partition.json'):
#        partition_dataset()
#
#    # open the partitioned data
#    with open('../data/LOCO_partition.json') as f:
#        data = json.load(f)
#
#    # get the data (i realize this is inefficient)
#    ct = split_data(data)
#    ct = ct[data_index]
#
#    # split the data into x and y vectors
#    xdata, ydata = x_y_split(ct)
#
#    # split the data into train and test
#    xtrain, xtest, ytrain, ytest = train_test_split(xdata, ydata,
#                                                    train_size = train_perc,
#                                                    test_size = test_perc)

    xtrain, xtest, ytrain, ytest  = read_data(index_to_seed(data_index))

    # vectorize the data
    vectorizer = CountVectorizer(analyzer = feat, ngram_range = (n1, n2))

    # convert data to vectorized form
    vec_xtrain = vectorizer.fit_transform(xtrain)
    vec_xtest = vectorizer.transform(xtest)

    # create SVM model and grid search
    svm = LinearSVC()
    param_grid = {'loss': ['squared_hinge'],
                      'random_state': [1291],
                      'C': [0.01,0.1,1]}

    model = GridSearchCV(svm, param_grid, n_jobs=16, verbose=1, cv=5)

    # print out our model parameters
    print(svm.get_params)

    # fit data
    svm.fit(vec_xtrain, ytrain)

    # predict with our svm model
    preds = svm.predict(vec_xtest)

    # check how our model did
    print(classification_report(ytest, preds, digits=4))


if __name__ == "__main__":
    main()
