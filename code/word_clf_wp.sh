for data in bf fe cc va pg; do  
  for e in bf fe cc va pg; do  
      python3 classify.py -d $data -e $e -m one -f wp -r 1,3 -o ./result_wp_one/
  done
done


for data in bf fe cc va pg; do  
  for e in bf; do  
      python3 classify.py -d $data -e $e -m merge -f wp -r 1,3 -o ./result_wp_mix/
  done
done
