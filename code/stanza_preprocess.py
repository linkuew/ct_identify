
from tqdm import tqdm
import stanza
import sys
import pandas as pd
import copy
'''
legacy code

'''
def pos_stanza(file):
    # combine all lines together
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse')
    with open(file+'.pos','w',encoding='utf-8') as o:
        for l in tqdm(open(file,'r',encoding='utf-8',errors='ignore').readlines()):
            doc =nlp(l)
            for s in doc.sentences:
                pos = ' '.join([word.xpos for word in s.words])
                o.write(pos+'\n')

def pos_word_stanza(file):
    # combine all lines together
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma')
    with open(file+'.pos.word','w',encoding='utf-8') as o:
        for l in tqdm(open(file,'r',encoding='utf-8',errors='ignore').readlines()):
            doc =nlp(l)
            for s in doc.sentences:
                pos = ' '.join([ word.text+'_'+word.xpos for word in s.words])
                o.write(pos+'\n')

'''
Extract Dep triples
'''

def get_dep_triple(seed,folder):
    df = pd.read_csv(folder+seed+'.csv')
    dep_triples = []
    for idx, val in tqdm(df['text'].iteritems()):
        doc = nlp(val)
        doc_deps = []
        for sent in doc.sentences:
            sent_deps = []
            for word in sent.words:
                if word.head == 0:
                    head_word = 'ROOT'
                else:
                    head_word = sent.words[word.head-1].text
                dep_triple = word.text+'_'+head_word+'_'+word.deprel
                sent_deps.append(dep_triple)
            doc_deps.append(' '.join(sent_deps))
        dep_triples.append(' '.join(doc_deps))
                
    df['dep_triple'] = dep_triples
    df.to_csv(folder+seed+'.dep.csv')

def get_word_pos(seed,folder):
    df = pd.read_csv(folder+seed+'.csv')
    pos_docs = []
    wp_docs = []
    for idx, val in tqdm(df['text'].iteritems()):
        doc = nlp(val)
        doc_sents = []
        doc_sents_wp = []
        for sent in doc.sentences:
            pos = ' '.join([word.xpos for word in sent.words])
            doc_sents.append(pos)
            wp = ' '.join([word.text+'_'+word.xpos for word in sent.words])
            doc_sents_wp.append(wp)            
        pos_docs.append(' '.join(doc_sents))
        wp_docs.append(' '.join(doc_sents_wp))
    
    df_copy = df.copy()

    df['pos'] = pos_docs
    df.to_csv(folder+seed+'.pos.csv')
    
    df_copy['word_pos'] = wp_docs
    df_copy.to_csv(folder+seed+'.wp.csv')    



nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma')

glob_dict = {'bf' : 'big.foot', 'fe' : 'flat.earth', 'cc' : 'climate', 'va' : 'vaccine', 'pg' : 'pizzagate'}
glob_dict_rev = {value:key for key, value in glob_dict.items()}
seeds = list(glob_dict_rev.keys())
acronyms = list(glob_dict.keys())

folder = '../data/train_'
for seed in seeds:
    get_word_pos(seed,folder)

folder = '../data/test_'
for seed in seeds:
    get_word_pos(seed,folder)

folder = '../data/val_'
for seed in seeds:
    get_word_pos(seed,folder)

folder = '../data/data_4/train_'
for seed in seeds:
    acronyms_copy = copy.deepcopy(acronyms)
    acronyms_copy.remove(glob_dict_rev[seed])
    new_seed = '_'.join(acronyms_copy)  
    get_word_pos(new_seed,folder)

folder = '../data/data_4/val_'
for seed in seeds:
    acronyms_copy = copy.deepcopy(acronyms)
    acronyms_copy.remove(glob_dict_rev[seed])
    new_seed = '_'.join(acronyms_copy)  
    get_word_pos(new_seed,folder)


