#!/usr/bin/env python
# coding: utf-8

# ### Welcome to your notebook
# 
# Learn how to write your own cells containing Markdown text and Python code.

# In[1]:


# If you want to plot graphics in your notebook, keep this %matplotlib command here before your imports
# %matplotlib notebook

# Import some necessary modules
import time
from collections import Counter, defaultdict
from random import shuffle

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.corpus import subjectivity as sub
from nltk.stem import WordNetLemmatizer 


# In[ ]:





# In[46]:


def divide_sets(labeled_sents):
    shuffle(labeled_sents)
    ratio = int(len(labeled_sents) * 0.8)
    train, test = labeled_sents[:ratio], labeled_sents[ratio:]
    return train, test


# In[91]:


labeled_sents = [(sent, label) for label in sub.categories() for sent in sub.sents(categories=label)]
trainset, testset = divide_sets(labeled_sents)


# In[92]:


print(trainset[0])
print(testset[0])


# In[93]:


lemtzer = WordNetLemmatizer()
stopws = stopwords.words('english')
fd = nltk.FreqDist(lemtzer.lemmatize(w) for sent, label in trainset for w in sent
                    if w.isalnum() and w not in stopws)
ftwords = list(fd)[:500]

def sent2ft(sent):
    features = {}
    lemmasent = [lemtzer.lemmatize(w.lower()) for w in sent]
    for w in ftwords:
        features['contains({})'.format(w)] = (w in lemmasent)
    return features

def data2ftset(sentset):
    return [(sent2ft(sent), label) for sent, label in sentset]


# In[94]:


print(ftwords[:10])


# In[95]:


train_ftset, test_ftset = data2ftset(trainset), data2ftset(testset)


# Transform the data (the tokenized sentences) into classifier recognizable formats, i.e. the features.
# 
# The feature here is whether the sentence contains each of the featured words (parameter ``ftwords``), which are picked from the 500 most frequent non-stop words in the training set.

# In[96]:


train_ftset[0]


# In[ ]:





# In[75]:





# In[131]:


clfNB = nltk.NaiveBayesClassifier.train(train_ftset)
print('NB Accuracy: {:4.2f}'.format(nltk.classify.accuracy(clfNB, test_ftset))) 
clfNB.show_most_informative_features(20)
    
# NBResults()


# In[132]:


def show_errcases(n_errcases, classifier, test_set):
    errlst, i = [], 0
    while len(errlst) <= n_errcases: # Get 10 wrongly classified sentences
        sent, goldlabel = test_set[i]
        sentft = sent2ft(sent)
        testlabel = classifier.classify(sentft)
        if testlabel != goldlabel:
            errlst.append([testlabel, sent, sentft])
        i += 1
        
    print('%s wrongly classified instances:' % n_errcases)
    print()
    
    for wronglabel, sent, sentft in errlst:
        print(sent)
        print('Wrongly classified as: %s' % wronglabel)
        informative_fts =  [(ft, sentft[ft]) for ft in sentft if sentft[ft] == True]
        print('Informative features:', informative_fts)
        print()
        
show_errcases(10, clfNB, testset)


# In[133]:


def classifier_label_list(classifier, test_ftset):
    testlst, goldlst = [], []
    for ft, lb in test_ftset:
        testlst.append(classifier.classify(ft))
        goldlst.append(lb)
    return testlst, goldlst

def show_cm(classifier, test_ftset):
    gold, test = classifier_label_list(classifier, test_ftset)
    cm = nltk.ConfusionMatrix(gold, test)
    print(cm.pretty_format(sort_by_count=True, show_percents=True))
    
show_cm(clfNB, test_ftset)


# In[ ]:




