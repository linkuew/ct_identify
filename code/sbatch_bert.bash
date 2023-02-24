#!/bin/bash

<<<<<<< HEAD
=======
#SBATCH --account r00018
>>>>>>> 208ab7423b54c6b954791fb8c1dcdd0a086abf80
#SBATCH -p dl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node p100:1
#SBATCH -o base_out_%j.out
#SBATCH -e base_error_%j.err
#SBATCH --mail-user=zuoytian@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -J ct_clf
#SBATCH --time=30:00:00

for data in bf; do  
  for e in fe; do  
    for p in 5; do 
        python bert_clf.py -d $data -e $e -m one -p 1 -l 1e-5 -b 5
    done
  done
done
 

