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
5. go to the code folder, *cd ./ct_identify/code*
6. to be updated, *git pull*
7. run different python scripts: classify.py, bert_clf.py, feature_selection.py
8. download all the result files from the server. If you use windows, try WinSCP, if you use Mac, try Cyberduck
9. upload the results to the onedrive shared folder

### Example command line

#### Parameters


-d, dataset to use

bf = bigfoot

fe = flat earth

cc = climate change

va = vaccines

pg = pizzagate

-e, dataset to do the evaluation

-f, feature set, either 'word', 'char', 'pos', 'dep', or 'wp' (word&pos, which is content word bleaching) this is for classy.py

-r, n-gram range for features, e.g. 1,3, this is for classy.py

-o, the path of the results output folder

-k, number of top features, use 'all' to get all features, this is for feature_selection.py


-s, selection function for features, either 'chi2' or 'mutual_info_classif', this is for feature_selection.py

-p, epoch for training the bert classifier, e.g. 3, this is for bert_clf.py

-r, learning rate for training the bert classifier, e.g. 1e-5, this is for bert_clf.py

-b, batch size for training the bert classifier, e.g. 32, this is for bert_clf.py


### Use Command line to run script: 

#### SVM classification (on CL server): 

!!! USE python3 in the cl server

*python3 classify.py -d bf -e fe -m merge -f word -r 1,3 -o ./result_svm_mix/*

#### Feature Selction (on CL server):

*python3 feature_selection.py -d bf -e fe -m merge -f word -r 1,3  -k 1000 -s chi2 -o ./result_feat_mix/*

#### BERT Classification (on Carbonate): 

*sbatch sbatch_bert.bash*

include the following line in the batch file:

*python bert_clf.py -d bf -e fe -m merge -p 5 -l 1e-5 -b 5 -o ./result_bert_mix/*


### Use bash script to run experiments for different parameters: 

*nohup bash <bash_script_name>*

add *nohup* will let the script not stop when the user logs out

for example, if you want to do experiments on all CTs with the one mode (train on one test on another) using word ngram do feature selection (chi2), try the following:

*nohup bash feat_sel.sh*

## roadmap (temp)

### features

try different feature combinations: 1) word ngram: done! 2) char ngram: done! 3) transformers: done! 4) dep triples (word, head, deprel): TODO

try different feature selection methods: 1) chi2 2) mutual information 3) select by models: logistic regression or random forest

### pipleline

1. train on one and test on the others (four)

2. train on four (same size) and test on one
