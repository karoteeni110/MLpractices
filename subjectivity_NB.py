# Supervised learning: subjectivity
import time
from collections import Counter, defaultdict
from random import shuffle

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.corpus import subjectivity as sub
from sacremoses import MosesDetokenizer

# Lemmatizing

# Filtering stop words 

# ~~~~~~~~~~

def sent2feature(sent):
    features = {}
    for w in ftwords:
        features['contains({})'.format(w)] = (w in sent)
    return features

stopws = stopwords.words('english')
fd = nltk.FreqDist(w.lower() for w in sub.words() if w not in stopws and w.isalpha())
ftwords = set(list(fd)[:500])

# print(sent2feature(sub.sents()[9]))

# data = [(sent2feature(sent), label) for label in sub.categories()
#                       for sent in sub.sents(categories=label)]
# shuffle(data)
# trainset, testset = data[:8000], data[8000:]

def get_ftset(sentset):
    return [(sent2feature(sent), label) for sent, label in sentset]

labeled_sents = [(sent, label) for label in sub.categories()
                    for sent in sub.sents(categories=label)]
shuffle(labeled_sents)
train_sents, test_sents = labeled_sents[:8000], labeled_sents[8000:]
trainset, testset = get_ftset(train_sents), get_ftset(test_sents)


# Result from NLTK built-in NB classifier
def nltkResults():
    # start = time.time()
    classifier = nltk.NaiveBayesClassifier.train(trainset)
    # print(nltk.classify.accuracy(classifier, testset))
    MIFs = classifier.most_informative_features(20)
    classifier.show_most_informative_features(5)
    # end = time.time()
    # print('Time:', end-start)
    errlst = []
    dt = MosesDetokenizer(lang='en')
    for sent,gold_l in test_sents:
        ft = sent2feature(sent)
        pred_l = classifier.classify(ft)
        probs = classifier.prob_classify(ft)
        if pred_l != gold_l:
            errlst.append([pred_l, sent, ft, probs])
    print('Errs:', len(errlst))
    ensi_20 = int(len(errlst) * 0.2)
    for pred_l, sent, ftset, probs in errlst[:ensi_20]:
        print('Wrong prediction: %s' % pred_l)
        # print(('Subj %f VS Obj %f') % (probs['subj'], probs['obj']))
        print(dt.detokenize(sent, return_str=True))
        for ft in ftset:
            if ft in MIFs:
                print('Informative feature: %s:%s' % (ft,ftset[ft]))
        print()

nltkResults()
