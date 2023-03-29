for data in bf fe cc va pg; do  
  for e in bf fe cc va pg; do  
      python3 classify.py -d $data -e $e -m one -f word -r 1,3 -y true -o ./result_one/
  done
done


for data in bf fe cc va pg; do  
  for e in bf; do  
      python3 classify.py -d $data -e $e -m merge -f word -r 1,3 -y true -o ./result_mix/
  done
done