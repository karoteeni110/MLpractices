"""https://universaldependencies.org/treebanks/fi_tdt/index.html"""
import csv, sys, math
from os import listdir

DATAHEADER = ['BODY','POS']

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
tag_to_logfreq = {"DET": logfreq(3), "NN": logfreq(4), "V": logfreq(2)}

def logfreq(freq):
    return int(math.log(freq))

def add_word_to_ix():
    word_to_ix = {}
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

def read_conllu(filename):
    """
    Read a list of sentences with POS labels from @sefilename. Each
    sentence is a tuple of elements:
    
    BODY      - List of tokens of the sentence.
    POS       - POS label for the sentence.


    """
    data = []

    with open(filename, encoding='utf-8') as sefile:
        pass
        # TODO: write code here
    return data

def read_ud_datasets(data_dir):
    print('Reading', data_dir)
    data = {}
    for data_set in ['test', 'dev', 'train']:
        data[data_set] = read_conllu("%s/tdt-ud-%s.conllu" % (data_dir,data_set)) 
    return data

if __name__=="__main__":
    # Check that we don't crash on reading.
    pass

