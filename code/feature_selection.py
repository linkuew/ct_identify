from library import *
import os, sys

# python classify.py -d bf -e fe -m merge -f word -r 1,3  -k 1000 -s chi2

def main():
    if len(sys.argv) < 10:
        print (len(sys.argv))
        usage(sys.argv[0])
        exit(1)

    seed, seed_eval, mode, feat, low, upp, num_feat, func, outpath = process_args(sys.argv[0])                                                                                                                                                                 


    print (seed)
    print (seed_eval)

    if mode == 'merge':
        # test on seed, train on a list without seed
        train, test, val = read_data_merge(seed,seed_eval)
    else:
        train, test, val  = read_data(seed,seed_eval)



    # find best features for this ct
    feature_selection(train, test, val, num_feat, low, upp, func, feat, seed, seed_eval, outpath, 10, False)

if __name__ == "__main__":
    main()
