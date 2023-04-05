'''
train clf for CT texts


'''

import os
import json
import sys
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from datasets import load_metric
from library import *


# load model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    

############################################################################
#
# This is the script for running classification using transformer architectures.
# Notice that currently, we only used the first 512 tokens for classification
#
# It is basically same with the svm ones, but you probably need to run it on Carbonate GPU nodes
# python bert_clf.py -d bf -e fe -m one -p 1 -l 1e-5 -b 5
# 
# I will add the batch script in src soon
# 
# After setting up the environment in Carbonate
# Run: sbatch sbatch_bert.bash
############################################################################

# Prepare the text inputs for the model
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, return_tensors = 'pt')

def compute_metrics(eval_pred):
    load_f1 = load_metric("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = load_f1.compute(predictions=predictions, references=labels, average = 'macro')["f1"]
    return {"f1": f1}

def main():

    if len(sys.argv) < 10:
        print (len(sys.argv))
        usage(sys.argv[0])
        exit(1)

    seed, seed_eval, mode, ep, lr, batch, outpath = process_args(sys.argv[0])

    print (seed)
    print (seed_eval)


    if mode == 'merge':
        # test on seed, train on a list without seed
        xtrain, ytrain, xtest,  ytest = read_data_merge(seed,seed_eval)
    else:
        xtrain, ytrain, xtest, ytest  = read_data(seed,seed_eval)

    
    train = pd.concat([xtrain, ytrain], axis=1)
    test = pd.concat([xtest, ytest], axis=1)


    train['label'] = train['label'].replace({'mainstream':0, 'conspiracy':1})
    test['label'] = test['label'].replace({'mainstream':0, 'conspiracy':1})

    tr_data, val_data = train_test_split(train, test_size = 0.1)


    tr_dataset = Dataset.from_pandas(tr_data)
    # val_dataset
    val_dataset = Dataset.from_pandas(val_data)
    # test dataset
    te_dataset = Dataset.from_pandas(test)



    # currently, we only use the first 512 tokens for the whole document
    tokenized_train = tr_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = te_dataset.map(preprocess_function, batched=True)

    

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir='./',
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        num_train_epochs=ep,
        weight_decay=0.01,
        save_strategy="epoch", 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        compute_metrics= compute_metrics,
        data_collator=data_collator,
    )


    trainer.train()
    trainer.evaluate()
    
    # Eval on test set
    predictions = trainer.predict(tokenized_test)
    preds = np.argmax(predictions.predictions, axis=-1)

    # dataset eval
    out_res = outpath+'res_tr_'+seed+'_te_'+seed_eval+'.csv'

    transform_labels = ['mainstream' if x == 0 else x for x in preds]
    transform_labels = ['conspiracy' if x == 1 else x for x in transform_labels]

    report = classification_report(ytest, transform_labels, digits=4, output_dict= True)
    print (report)
    df = pd.DataFrame(report).transpose()

    df.to_csv(out_res)


if __name__ == "__main__":
    main()
