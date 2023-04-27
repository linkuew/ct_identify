'''
train clf for CT texts


'''

import os
import sys
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from transformers import BartForSequenceClassification, BartTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from datasets import load_metric
from torch.utils.data import Dataset, DataLoader
from library import *
import torch
from transformers import AdamW



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
class ClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def evaluate(model, test_loader):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            _, pred = torch.max(logits, dim=1)
            predictions.extend(pred.tolist())
            true_labels.extend(batch["labels"].tolist())

    return predictions, true_labels


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

    train_encodings = tokenizer(train['text'].tolist(), truncation=True, padding=True)

    train_dataset = ClassificationDataset(train_encodings, tr['label'].tolist())
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)



    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    
    test_encodings = tokenizer(test['text'].tolist(), truncation=True, padding=True)

    test_dataset = ClassificationDataset(test_encodings, test['label'].tolist())
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    predictions, true_labels = evaluate(model, test_loader)
    


    if not os.path.exists(outpath):
        os.makedirs(outpath)
    out_res = outpath+'res_tr_'+seed+'_te_'+seed_eval+'.csv'


    transform_labels = ['mainstream' if x == 0 else x for x in predictions]
    transform_labels = ['conspiracy' if x == 1 else x for x in transform_labels]

    report = classification_report(true_labels, transform_labels, digits=4, output_dict= True)
    print (report)
    df = pd.DataFrame(report).transpose()

    df.to_csv(out_res)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
