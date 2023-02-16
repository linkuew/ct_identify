# ct_identify
Code-base and data to build models to identify conspiracy theories.


## feature selection
use scikitlearn
https://scikit-learn.org/stable/modules/feature_selection.html



## experiment setup

1. run all experiments on the server (cl servwer)
2. create a virtual environment
3. install required libraries
4. git clone this repo
5. to be updated, git pull
6. run different python scripts: classify.py, bert_clf.py, feature_selection.py


## roadmap (temp)

### features

try different feature combinations: 1) word ngram: done! 2) char ngram: done! 3) transformers: done! 4) dep triples (word, head, deprel): TODO

try different feature selection methods: 1) chi2 2) mutual information 3) select by models: logistic regression or random forest

### pipleline

1. train on one and test on the others (four)

2. train on four (same size) and test on one
