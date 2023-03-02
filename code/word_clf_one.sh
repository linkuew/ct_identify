for data in bf fe cc va pg; do  
  for e in bf fe cc va pg; do  
      python3 classify.py -d $data -e $e -m one -f word -r 1,3
  done
done
