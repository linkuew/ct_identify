# ct_identify
Code-base and data to build models to identify conspiracy theories.


## feature selection
use scikitlearn
https://scikit-learn.org/stable/modules/feature_selection.html



## experiment setup

1. run all experiments on the server (cl servwer)
2. create a virtual environment
3. install scikitlearn
4. upload files (loco data)
5. run python scripts (nohup python your_script.py)


## roadmap (temp)

### features

try different feature combinations: 1) word ngram 2) char ngram 3) transformers 4) dep triples (word, head, deprel)

try different feature selection methods: 1) chi2 2) mutual information 3) select by models: logistic regression or random forest

### pipleline

1. train on one and test on the others (four)

2. train on four (same size) and test on one
