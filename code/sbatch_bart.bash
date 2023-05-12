#!/bin/bash

#SBATCH -A general
#SBATCH -p gpu
#SBATCH --mail-user=mattfort@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --time=0-05:00:00
#SBATCH --mem=50gb
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --job-name=ct_%j
#SBATCH -o ./../../out/base_out_%j.out
#SBATCH -e ./../../err/base_error_%j.err

# d, dataset to use
# bf = bigfoot
# fe = flat earth
# cc = climate change
# va = vaccines
# pg = pizzagate
# f, feature set, either 'word' or 'char'
# t, train percent, test percent, e.g.: 70,30
# r, n-gram range for features, e.g. 1,3
# -k, number of top features, choose the target number of features here, e.g. 1000
# -s, selection function for features, either 'chi2' or 'mutual_info_classif'
# -ep, epoch
# -lr, learning rate

module load python/gpu/3.10.10

declare -a cts=("bf" "fe" "cc" "va" "pg")

# set variables so it's easier to edit

# training set
tr="tmp"

# testing set
#te="bf"

# feature set, (dep, pos, wp, word, char, tp)
feature="tp"

# epochs
epochs=5

# learning rate
lr="1e-5"

# batch size
batch=2

# outpath
path="~/results/"

# mode (merge, ??)
mode="merge"


# test/train percentage
#trte_split="70,30"

# ngram range
#ngrams="1,3"

#selection function (chi2, mutual_info_classif)
#selection="chi2"

##
# setup to run through things all together
##
#for tr in "${cts{@}}"; do
        #-d $tr \
for te in "${cts[@]}"; do
        python bert_clf.py \
                -e $te \
                -m $mode \
                -p $epochs \
                -l $lr \
                -b $batch \
                -o $path \
                -f $feature
done
#done
