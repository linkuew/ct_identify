for data in bf fe cc va pg; do  
  for e in bf fe cc va pg; do  
    for k in 500 1000 2000 3000; do 
        python3 feature_selection.py -d $data -e $e -m one -f word -r 1,3 -k $k -s chi2
    done
  done
done