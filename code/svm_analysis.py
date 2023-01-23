import json
import random
import nltk
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import LinearSVC
from tqdm import tqdm
lemmatizer = WordNetLemmatizer()

# Run these lines the first time you go through this notebook
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
nltk.download('wordnet')

### Run this line **only** if you are in Colab and need to get the data
# !git clone https://github.com/linkuew/ling715_project.git

random.seed(42)

##
# Split the data into the different areas which we want to test on
#
# input: entire corpus
#
# returns: subcorpora
##
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

    return [bigfoot, climate, flat, pizza, vaccine]

##
# A helper function for the conspiracy select, this returns the text entry from the corpus
#
# input: a corpus
#
# output: a breakdown of the input corpus into text (X) and label (Y) - note that 1 marks conspiracy and 0 marks non-conspiracy
##
def x_y_split(data):
    X = []
    Y = []

    for entry in data:
        if entry['subcorpus'] == 'conspiracy':
            X.append(entry['txt'])
            Y.append(1)
        else:
            X.append(entry['txt'])
            Y.append(0)

    return X, Y

##
# How we preprocess our data/create our feature sets will probably have most impact, this is a place for that to happen
#
# input: corpus text
#
# returns: modified corpus text
##
def preprocess(data):
    update = []
    for entry in data:
        word_data = word_tokenize(entry)
        # print(word_data)
        lemmatized = ' '.join(lemmatizer.lemmatize(x) for x in word_data)
        update.append(lemmatized.lower())
    return update

##
# Returns a combination of the input conspiracy corpora, and shuffles the input as well.
#
# input: sequence of tuples (corpus, percent), (corpus, percent),...
#
# returns: X and Y, such that X and Y are the percentages of the corpora given in the input
#
# note: this does *not* ensure balance between the conspiracy/non-conspiracy elements
# within the partition
##
def conspiracy_select(*args):
    newX = []
    newY = []

    for arg in args:
        # get text from a corpus
        tmpX, tmpY = x_y_split(arg[0])

        # randomize the order of the texts and labels
        tmpX, tmpY = randomize(tmpX, tmpY)

        # partition the texts and labels according to the percentage
        tmplen = int(arg[1] * len(tmpX))
        tmpX = tmpX[:tmplen]
        tmpY = tmpY[:tmplen]

        # update our combined texts and labels
        newX += tmpX
        newY += tmpY

    return newX, newY

##
# Randomizes X and Y pairwise data
#
# input: X and Y lists
#
# returns: X and Y lists pairwise shuffled
#
##
def randomize(X, Y):
    tmp = list(zip(X, Y))
    random.shuffle(tmp)
    tmpX, tmpY = zip(*tmp)
    return list(tmpX), list(tmpY)

##
# Returns xtrain, xtest, ytrain, and ytest by letting one CT be the xtrain and ytrain, and the other CT be the xtest
# and ytest
#
# input: two conspiracy theories where CT1 -> training, CT2 -> testing
#
# returns xtrain, xtest, ytrain, and ytest
##
def custom_test_train_split(CT1, CT2):
    xtrain, ytrain = x_y_split(CT1)
    xtest, ytest = x_y_split(CT2)

    # uncomment this if we want to randomize the pairwise order of texts and labels
    # xtrain, ytrain = randomize(xtrain, ytrain)
    # xtest, ytest = randomize(xtest, ytest)

    return xtrain, xtest, ytrain, ytest

##
# Transforms a set of documents into POS tags
#
# input: an list of text documents
#
# returns: a list in the same order of documents transformed into POS tags
##
def pos_tags(X):
    newX = []
    for entry in X:
        tmp = []
        for _, tag in pos_tag(word_tokenize(entry)):
            tmp.append(tag)
        newX.append(tmp)

    return newX

##
# Used with CountVectorizer with POS tags to just return the POS tag. CountVectorizer assumes it's taking in 'text'
# but we know it's taking in POS tags, all we want are the counts of these things, so we just return the POS tag we
# already have
##
def identity_tokenizer(text):
    return text

with open('../data/LOCO_partition.json') as f:
        data = json.load(f)

#bigfoot, climate, flat, pizza, vaccine
ct = split_data(data)

for i in tqdm(range(len(ct))):
    xtrain, ytrain = x_y_split(ct[i])
    # xtrain = pos_tags(xtrain)

    ctvec = CountVectorizer(tokenizer = word_tokenize, analyzer = 'char', ngram_range = (3,5), max_features = 5000)

    # ctvec = CountVectorizer(tokenizer = identity_tokenizer, max_features = 5000, lowercase = False)
    vecxtrain = ctvec.fit_transform(xtrain)

    SVM = LinearSVC(max_iter=100000)
    SVM.fit(vecxtrain, ytrain)

    for j in range(len(ct)):
        xtest, ytest = x_y_split(ct[j])
        # xtest = pos_tag(xtest)

        vecxtest = ctvec.transform(xtest)

        SVMpred = SVM.predict(vecxtest)

        pre = precision_score(SVMpred, ytest)
        rec = recall_score(SVMpred, ytest)
        f1 = f1_score(SVMpred, ytest)
        print('Result for ct['+str(i)+'] tested on ct['+str(j)+']')
        print(f"{pre*100:.2f}|{rec*100:.2f}|{f1*100:.2f}")

bigX, bigY = x_y_split(bigfoot)
# cliX, cliY = x_y_split(climate)
# X, Y = conspiracy_select((bigfoot, 1), (vaccine, 1), (flat, 1), (pizza, 1), (climate, 1))
# xtrain, xtest, ytrain, ytest = custom_test_train_split(bigfoot, vaccine)

# These lines will take a while! It's pretty expensive to POS tag the entire corpora
# xtrain = pos_tags(xtrain)
# xtest = pos_tags(xtest)

xtrain, xtest, ytrain, ytest = model_selection.train_test_split(bigX, bigY, test_size = 0.3)

# For POS tag feature set runs:
# ctvec = CountVectorizer(tokenizer = identity_tokenizer, max_features = 5000, lowercase = False)

# For the original runs:
ctvec = CountVectorizer(analyzer = 'word', tokenizer = word_tokenize, max_features = 5000)

vecxtrain = ctvec.fit_transform(xtrain)

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')

SVM.fit(vecxtrain, ytrain)

vecxtest = ctvec.transform(xtest)

SVMpred = SVM.predict(vecxtest)

print("SVM Accuracy -> ", accuracy_score(SVMpred, ytest) * 100)
