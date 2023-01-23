import json
import nltk
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

## Check to see if data contains are repeats since I'm not sure
## this was taken care of
def check_dups(data):
    doc_ids = []
    for entry in data:
        doc_ids.append(entry['doc_id'])
    if len(doc_ids) != len(set(doc_ids)):
        print('there is a dup!')
    else:
        print('no dup!')


# Split the data into the different areas which we want to test on
def split_data(data):
    vaccine = []
    bigfoot = []
    flat = []
    pizza = []
    climate = []
    for entry in data:
        if entry['seeds'].__contains__('big.foot'):
            bigfoot.append(entry)
        if entry['seeds'].__contains__('vaccine'):
            vaccine.append(entry)
        if entry['seeds'].__contains__('flat.earth'):
            flat.append(entry)
        if entry['seeds'].__contains__('pizzagate'):
            pizza.append(entry)
        if entry['seeds'].__contains__('climate'):
            climate.append(entry)

    return bigfoot, vaccine, flat, pizza, climate

# Find out how many elements in the corpus are conspiratorial
def polarity_analysis(topic):
    tmp_true = 0
    tmp_false = 0

    for i in range(len(topic)):
        if topic[i]['subcorpus'] == 'conspiracy':
            tmp_true += 1a
        else:
            tmp_false += 1
    return tmp_false, tmp_true

# does what it says on the can
def avg_text_length(topic):
    total = 0
    for i in range(len(topic)):
        total += topic[i]['txt_nwords']

    return float(total) / len(topic)

# does what it says on the can
def avg_sent_num(topic):
    total = 0
    for i in range(len(topic)):
        total += topic[i]['txt_nsentences']

    return float(total) / len(topic)

def avg_share_comment_react(topic):
    share = 0
    comment = 0
    react = 0
    for i in range(len(topic)):
        share += topic[i]['FB_shares']
        comment += topic[i]['FB_comments']
        react += topic[i]['FB_reactions']

    return (float(share) / len(topic), \
            float(comment) / len(topic), \
            float(react) / len(topic))

with open('../data/LOCO_partition.json') as f:
        data = json.load(f)

check_dups(data)

for key, _ in data[0].items():
    print(key)

for key, value in data[0].items():
    print(key, '=', value, '\n')

bigfoot, vaccine, flat, pizza, climate = split_data(data)

print(len(bigfoot),len(vaccine),len(flat),len(pizza),len(climate))

check_dups(bigfoot)
check_dups(vaccine)
check_dups(flat)
check_dups(pizza)
check_dups(climate)

print(polarity_analysis(bigfoot))
print(avg_text_length(bigfoot))
print(avg_sent_num(bigfoot))
print(avg_share_comment_react(bigfoot))

print(polarity_analysis(vaccine))
print(avg_text_length(vaccine))
print(avg_sent_num(vaccine))

print(polarity_analysis(flat))
print(avg_text_length(flat))
print(avg_sent_num(flat))

print(polarity_analysis(pizza))
print(avg_text_length(pizza))
print(avg_sent_num(pizza))

print(polarity_analysis(pizza))
print(avg_text_length(pizza))
print(avg_sent_num(pizza))

vocab = []
for i in range(len(climate)):
    vocab.append(climate[i]['txt'])

tfIdfVectorizer=TfidfVectorizer(use_idf=True)
tfIdf = tfIdfVectorizer.fit_transform(vocab)
df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print(df.head(25))


lsa = TruncatedSVD(algorithm='arpack').fit(tfIdf)

print(get_model_topics(lsa, tfIdf, lsa_topics))


def get_model_topics(model, vectorizer, topics, n_top_words=20):
    word_dict = {}
    feature_names = vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        word_dict[topics[topic_idx]] = top_features

    return pd.DataFrame(word_dict)
