import os
import json
import sys
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer

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


def main():
    if len(sys.argv) != 9:
        usage(sys.argv[0])
        exit(1)

    data_index, train_perc, test_perc, feat, n1, n2 = process_args(sys.argv[0])

    # check to make sure the LOCO partition exists
    if not os.path.exists('../data/LOCO_partition.json'):
        partition_dataset()

    # open the partitioned data
    with open('../data/LOCO_partition.json') as f:
        data = json.load(f)

    # get the data (i realize this is inefficient)
    ct = split_data(data)
    ct = ct[data_index]

    # split the data into x and y vectors
    xdata, ydata = x_y_split(ct)

    # split the data into train and test
    xtrain, xtest, ytrain, ytest = train_test_split(xdata, ydata,
                                                    train_size = train_perc,
                                                    test_size = test_perc)

    # vectorize the data
    vectorizer = CountVectorizer(analyzer = feat, ngram_range = (n1, n2))

    # convert data to vectorized form
    vec_xtrain = vectorizer.fit_transform(xtrain)
    vec_xtest = vectorizer.transform(xtest)

    # create SVM model
    svm = LinearSVC(C = 1.0)

    # print out our model parameters
    print(svm.get_params)

    # fit data
    svm.fit(vec_xtrain, ytrain)

    # predict with our svm model
    preds = svm.predict(vec_xtest)

    # check how our model did
    print("SVM accuracy: ", accuracy_score(preds, ytest) * 100)


if __name__ == "__main__":
    main()