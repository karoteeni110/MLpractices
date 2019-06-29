# Supervised learning: subjectivity
from nltk.corpus import subjectivity as sub
from nltk.corpus import stopwords
import nltk
import numpy as np
from random import shuffle

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
ftwords = set(list(fd)[:2000])

# print(sent2feature(sub.sents()[9]))

data = [(sent2feature(sent), cate) for cate in sub.categories() 
                      for sent in sub.sents(categories=cate)]
shuffle(data)
trainset, testset = data[:8000], data[8000:]

class myNBClassifier():
    """
    From prior probability:
    P(label)
    P(feature|label)

    To posterior probability:
    P(label|feature)

    And decide label based on
    """
    def __init__(self, label_probdist, feature_probdist):
        pass

    def train(self):
        pass

    def acc(self, test_set):
        pass
    

# Result from NLTK built-in NB classifier
classifier = nltk.NaiveBayesClassifier.train(trainset)
print(nltk.classify.accuracy(classifier, testset))
classifier.show_most_informative_features(5)