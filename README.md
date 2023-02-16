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

### Example command line

#### Parameters


-d, dataset to use

bf = bigfoot

fe = flat earth

cc = climate change

va = vaccines

pg = pizzagate

-e, dataset to do the evaluation

-f, feature set, either 'word' or 'char', this is for classy.py

-r, n-gram range for features, e.g. 1,3, this is for classy.py

-k, number of top features, use 'all' to get all features, this is for feature_selection.py


-s, selection function for features, either 'chi2' or 'mutual_info_classif', this is for feature_selection.py

-p, epoch for training the bert classifier, e.g. 3, this is for bert_clf.py

-r, learning rate for training the bert classifier, e.g. 1e-5, this is for bert_clf.py

-b, batch size for training the bert classifier, e.g. 32, this is for bert_clf.py


#### SVM classification (on CL server): 



*python3 classify.py -d bf -e fe -m merge -f word -r 1,3*

#### Feature Selction (on CL server):

*python3 feature_selection.py -d bf -e fe -m merge -f word -r 1,3  -k 1000 -s chi2*

#### BERT Classification (on Carbonate): 

*python bert_clf.py -d bf -e fe -m merge -p 5 -l 1e-5 -b 5*



## roadmap (temp)

### features

try different feature combinations: 1) word ngram: done! 2) char ngram: done! 3) transformers: done! 4) dep triples (word, head, deprel): TODO

try different feature selection methods: 1) chi2 2) mutual information 3) select by models: logistic regression or random forest

### pipleline

1. train on one and test on the others (four)

2. train on four (same size) and test on one
