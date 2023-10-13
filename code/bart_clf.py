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
from transformers import BartForSequenceClassification, BartTokenizer
from datasets import load_metric
from library import *
import time
import torch
from transformers import AdamW
import time

# load model
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForSequenceClassification.from_pretrained(model_name, num_labels=2)

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
def preprocess_function(text):
    tokenized_text = tokenizer(text)

    if len(tokenized_text['input_ids']) <= 512:
        tokenized_text = tokenizer(text, return_tensors="pt", padding="max_length", max_length=1024)
     #   print("less than 512  = ", end=" ")
     #   print(tokenized_text['input_ids'].size())
     #   print(type(tokenized_text['input_ids']))
     #   print(tokenized_text)
    else:
        first512_words = tokenized_text['input_ids'][:512]
        last512_words = tokenized_text['input_ids'][-512:]
        totalwords = first512_words + last512_words

        first512_attention = tokenized_text['attention_mask'][:512]
        last512_attention = tokenized_text['attention_mask'][-512:]
        totalattention = first512_attention + last512_attention

        #tokenized_text['input_ids'] = totalwords
        #tokenized_text['attention_mask'] = totalattention

        tokenized_text['input_ids'] = torch.LongTensor([totalwords])
        tokenized_text['attention_mask'] = torch.LongTensor([totalattention])
        #print(type(tokenized_text['input_ids']))

        #print("after combining =", end=" ")
        #print(tokenized_text['input_ids'].size())
    tokenized_text['input_ids'] = tokenized_text['input_ids'].squeeze(0)
    tokenized_text['attention_mask'] = tokenized_text['attention_mask'].squeeze(0)

    return tokenized_text


class ClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    tokenized_text['input_ids'] = torch.LongTensor([totalwords]).squeeze(0)
    tokenized_text['attention_mask'] = torch.LongTensor([totalattention]).squeeze(0)

#    tokenized_text['input_ids'] = tokenized_text['input_ids'].squeeze(0)
#    tokenized_text['attention_mask'] = tokenized_text['attention_mask'].squeeze(0)

    return tokenized_text

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

    if mode == 'merge': # test on seed, train on a list without seed
        tr, te, val = read_data_merge(seed,seed_eval)
    else:
        tr, te, val  = read_data(seed,seed_eval)

    #train = pd.concat([xtrain, ytrain], axis=1)
    #test = pd.concat([xtest, ytest], axis=1)

    tr['label'] = tr['label'].replace({'mainstream':0, 'conspiracy':1})
    te['label'] = te['label'].replace({'mainstream':0, 'conspiracy':1})
    val['label'] = val['label'].replace({'mainstream':0, 'conspiracy':1})

    tokenized_train = tr['text'].map(preprocess_function)
    tokenized_val = val['text'].map(preprocess_function)
    tokenized_test = te['text'].map(preprocess_function)

    train_dataset = ClassificationDataset(tokenized_train, tr['label'].tolist())
    train_loader = DataLoader(tokenized_train, batch_size=batch, shuffle=True)

    test_dataset = ClassificationDataset(tokenized_test, te['label'].tolist())
    test_loader = DataLoader(tokenized_test, batch_size=batch, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(1):
        model.train()
        for batch in train_loader:
            print("going through a batch")
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    predictions, true_labels = evaluate(model, test_loader)

    if not os.path.exists(outpath):
        os.makedirs(outpath)
    out_res = outpath+'res_tr_'+seed+'_te_'+seed_eval+'_time_'+str(time.time())+'.csv'

    print(transform_labels)
    print()
    print(te['label'])

    report = classification_report(te['label'], transform_labels, digits=4, output_dict= True)
    print (report)

    df = pd.DataFrame(report).transpose()

    df.to_csv(out_res)

if __name__ == "__main__":
    main()
