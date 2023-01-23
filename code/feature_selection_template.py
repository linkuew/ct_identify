from sklearn.feature_selection import SelectKBest,mutual_info_classif,chi2,f_regression,f_classif
import scipy.sparse
import scipy.io
import numpy as np
import pandas as pd


def feature_selection(filepath, kfeature, n1=1,n2=5,min=10,bi=False):
    train_file_path = filepath+'.tr'
    test_file_path = filepath+'.te'

    train_set = pd.read_csv(train_file_path, sep="\t",header=None)
    x_train = train_set.iloc[:, 0].values
    y_train = train_set.iloc[:, 1].values
    test_set = pd.read_csv(test_file_path, sep="\t",header=None)
    x_test = test_set.iloc[:, 0].values

    vectorizer_word = CountVectorizer(analyzer="word",
                                ngram_range=(n1, n2),
                                min_df=min,
                                binary=bi,
                                token_pattern=r'\b\w+\b')

    train_bool_matrix = vectorizer_word.fit_transform(x_train)
    test_bool_matrix = vectorizer_word.transform(x_test)
    features = vectorizer_word.get_feature_names()


    bestfeatures = SelectKBest(chi2, k=kfeature)
    train_new = bestfeatures.fit_transform(train_bool_matrix, y_train)
    print (train_new.shape)
    test_new =bestfeatures.transform(test_bool_matrix) 
    print (test_new.shape)

    fit = bestfeatures.fit(train_bool_matrix,y_train)
    dfscores = pd.DataFrame(fit.scores_)
#     save the training and testing data
    scipy.io.mmwrite("./tr-data/train_bleach_sel"+str(kfeature)+".mtx", train_new)
    scipy.io.mmwrite("./te-data/test_bleach_sel"+str(kfeature)+".mtx", test_new)
#     save the features and their scores
    dfcolumns = pd.DataFrame(features)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    rank = featureScores.nlargest(500,'Score')
    rank.to_csv('./feature_selection/'+str(i+1)+"sel"+str(kfeature)+".txt")

