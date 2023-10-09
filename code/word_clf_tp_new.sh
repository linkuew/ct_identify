#!/bin/bash

# Loop over the values for i
for i in 10 15 20; do
    echo "Running for n_topics = $i"

    # Loop over the values for data
    for data in bf fe cc va pg; do
        echo "Processing data: $data"

        # Loop over the values for e
        for e in bf fe cc va pg; do
            echo "Evaluating: $e"

            # Run the topic_word_bleach.py script
            python3 topic_word_bleach.py --n_topics $i
            # Check if the script executed successfully
            if [ $? -ne 0 ]; then
                echo "Error during topic_word_bleach for n_topics=$i, data=$data, e=$e"
                exit 1
            fi

            # Run the classify.py script
            python3 classify.py -d $data -e $e -m one -f tp$i -r 1,3 -o ./result_tp$i/
            # Check if the script executed successfully
            if [ $? -ne 0 ]; then
                echo "Error during classify for n_topics=$i, data=$data, e=$e"
                exit 1
            fi

        done
    done
done


