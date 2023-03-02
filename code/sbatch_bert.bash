#!/bin/bash

#SBATCH --mail-user=PLACES USER EMAIL ADDRESS HERE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-30:00:00
#SBATCH --mem=50gb
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --job-name=ct_flf
#SBATCH -o base_out_%j.out
#SBATCH -e base_error_%j.err

## This is necessary for some, uncomment if you need it
# module load python

# testing array with the data
declare -a cts=("bf" "fe" "cc" "va" "pg")

# This is our training set
tr="pg"

# iterate through each CT, train on the same one but test on all the others
for te in "${cts[@]}"; do
        if [ "$tr" != "$te" ]; then
                python bert_clf.py -d $tr -e $te -m one -p 1 -l 1e-5 -b 5
        fi
done


