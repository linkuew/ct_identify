from library import *
import os, sys

# python classify.py -d bf -e fe -m merge -f word -r 1,3  -k 1000 -s chi2

def main():
    if len(sys.argv) < 8:
        print ('Wrong number of params')
        print (len(sys.argv))
        usage(sys.argv[0])
        exit(1)

    seed, seed_eval, mode, feat, low, upp, num_feat, func = process_args(sys.argv[0])                                                                                                                                                                 


    print (seed)
    print (seed_eval)

    if mode == 'merge':
        # test on seed, train on a list without seed
        xtrain, ytrain, xtest,  ytest = read_data_merge(seed,seed_eval)
    else:
        xtrain, ytrain, xtest, ytest  = read_data(seed,seed_eval)



    # find best features for this ct
    feature_selection(xtrain, ytrain,xtest,  ytest, num_feat, low, upp, func, feat, seed, seed_eval, 10, False)

if __name__ == "__main__":
    main()
