# ct_identify
Code-base and data to build models to identify conspiracy theories.


## feature selection
use scikitlearn
https://scikit-learn.org/stable/modules/feature_selection.html



## experiment setup

1. run all experiments on the server (cl servwer) ssh yourusername@cl.indiana.edu
2. create a virtual environment (optional)
3. install required libraries (if it shows some library is in need, just type *pip3 install packagename*)
4. git clone this repo to the cl server:  *git clone https://github.com/linkuew/ct_identify.git*
5. to be updated, *git pull*
6. run different python scripts: classify.py, bert_clf.py, feature_selection.py

example command line

SVM classification: *python classify.py -d bf -e fe -m merge -f word -r 1,3*

Feature Selction: **

BERT Classification (on Carbonate): *python bert_clf.py -d bf -e fe -m merge -p 1 -l 1e-5 -b 5*



## roadmap (temp)

### features

try different feature combinations: 1) word ngram: done! 2) char ngram: done! 3) transformers: done! 4) dep triples (word, head, deprel): TODO

try different feature selection methods: 1) chi2 2) mutual information 3) select by models: logistic regression or random forest

### pipleline

1. train on one and test on the others (four)

2. train on four (same size) and test on one
