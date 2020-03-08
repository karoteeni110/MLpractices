"""https://universaldependencies.org/treebanks/fi_tdt/index.html"""
import csv
import sys

import nltk

DATAHEADER = ['BODY','POS']

word_to_ix = dict()
tag_to_ix = dict()

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
    data = {}
    for data_set in ["fi_tdt-ud-dev.conllu","fi_tdt-ud-test.conllu","fi_tdt-ud-train.conllu"]:
        data[data_set] = read_conllu("%s/%s.txt" % (data_dir,data_set)) 
    return data

if __name__=="__main__":
    # Check that we don't crash on reading.
    pass

