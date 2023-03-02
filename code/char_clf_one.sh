for data in bf fe cc va pg; do  
  for e in bf fe cc va pg; do  
      python3 classify.py -d $data -e $e -m one -f char -r 2,7
  done
done
