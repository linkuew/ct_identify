for data in bf fe cc va pg; do  
  for e in bf; do  
      python3 classify.py -d bf -e fe -m merge -f word -r 1,3
  done
done