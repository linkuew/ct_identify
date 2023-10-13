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


# load model
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForSequenceClassification.from_pretrained(model_name, num_labels=2)

#model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
#
#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


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
def preprocess_function(dataset):
    tokenized_text = tokenizer(dataset['text'])

    if len(tokenized_text['input_ids']) <= 512:
        tokenized_text = tokenizer(dataset['text'], padding="max_length", max_length=513)
        words = []
        for entry in tokenized_text['input_ids']:
            if entry == 2:
                pass
            else:
                words.append(entry)

        totalwords = words + tokenized_text['input_ids'][-512:]

        totalattention = tokenized_text['attention_mask'][-512:] \
                        + tokenized_text['attention_mask'][-512:]

    else:
        first512_words = tokenized_text['input_ids'][:512]
        last512_words = tokenized_text['input_ids'][-512:]
        totalwords = first512_words + last512_words

        first512_attention = tokenized_text['attention_mask'][:512]
        last512_attention = tokenized_text['attention_mask'][-512:]
        totalattention = first512_attention + last512_attention

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

    tr['label'] = tr['label'].replace({'mainstream':0, 'conspiracy':1})
    te['label'] = te['label'].replace({'mainstream':0, 'conspiracy':1})
    val['label'] = val['label'].replace({'mainstream':0, 'conspiracy':1})

    tr_dataset = Dataset.from_pandas(tr)
    val_dataset = Dataset.from_pandas(val)
    te_dataset = Dataset.from_pandas(te)

    train_tokenized = tr_dataset.map(preprocess_function)
    val_tokenized = val_dataset.map(preprocess_function)
    test_tokenized = te_dataset.map(preprocess_function)

#    print(train_tokenized)
#    print(val_tokenized)
#    print(test_tokenized)
#    print()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir='./',
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=1,
        num_train_epochs=ep,
        weight_decay=0.01,
        save_strategy="epoch",
        label_names=["mainstream", "conspiracy"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        #tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.evaluate()

    # Eval on test set
    predictions = trainer.predict(test_tokenized)

#    print(len(predictions))
#    print(type(predictions.predictions))
#
#
#    print(type(predictions.predictions[0]))
#    print(len(predictions.predictions[0]))
#    print(predictions.predictions[0])
#    print()
#    print(type(predictions.predictions[1]))
#    print(len(predictions.predictions[1]))
#    print(predictions.predictions[1])
#
#    print()
#    print(type(predictions.predictions[2]))
#    print(len(predictions.predictions[2]))
#    print(predictions.predictions[2])

    preds = np.argmax(predictions.predictions[1], axis=-1)

    # dataset eval
    out_res = outpath+'res_tr_'+seed+'_te_'+seed_eval+'_epochs_'+str(ep)+'_time_'+str(time.time())+'.csv'

    transform_labels = ['mainstream' if x == 0 else x for x in preds]
    transform_labels = ['conspiracy' if x == 1 else x for x in transform_labels]

    te['label'] = te['label'].replace({0:'mainstream', 1:'conspiracy'})

    print(transform_labels)
    print()
    print(te['label'])

    report = classification_report(te['label'], transform_labels, digits=4, output_dict= True)
    print (report)

    df = pd.DataFrame(report).transpose()

    df.to_csv(out_res)

if __name__ == "__main__":
    main()
