import os
import json
import sys
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV

from library import *


def main():
    if len(sys.argv) < 10:
        print (len(sys.argv))
        usage(sys.argv[0])
        exit(1)

    seed, seed_eval, mode, feat, n1, n2,outpath = process_args(sys.argv[0])

    print (seed)
    print (seed_eval)

    top_features = 20

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

    # create SVM model and grid search
    svc = LinearSVC(loss= 'squared_hinge', random_state=1291,C=0.01)

    # fit data
    svc.fit(vec_xtrain, tr['label'])


    coef = svc.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]

    feature_names = vectorizer.get_feature_names()
    feature_names = np.array(feature_names)

    top_pos = feature_names[top_positive_coefficients]
    top_pos_coef = coef[top_positive_coefficients]

    pos_pairs = []
    for i in range(top_features):
        pos_pairs.append((top_pos[i],top_pos_coef[i]))

    top_neg = feature_names[top_negative_coefficients]
    top_neg_coef = coef[top_negative_coefficients]

    neg_pairs = []
    for i in range(top_features):
        neg_pairs.append((top_neg[i],top_neg_coef[i]))


    with open(outpath+'top_coefficient.csv','a') as out_param:
        out_param.write(seed+'\t')
        out_param.write('mainstream'+'\t')
        out_param.write(','.join(map(str, pos_pairs))+'\n')

        out_param.write(seed+'\t')
        out_param.write('conspiracy'+'\t')
        out_param.write(','.join(map(str, neg_pairs))+'\n')

    
if __name__ == "__main__":
    main()
