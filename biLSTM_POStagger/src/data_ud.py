"""https://universaldependencies.org/treebanks/fi_tdt/index.html"""
import csv, sys, math
from os import listdir
from collections import Counter, defaultdict

TRAIN_FPATH = '/Volumes/Valar Morghulis/ud_data/fi_tdt-ud-train.conllu.txt'
TEST_FPATH = '/Volumes/Valar Morghulis/ud_data/fi_tdt-ud-test.conllu.txt'
DEV_FPATH = '/Volumes/Valar Morghulis/ud_data/fi_tdt-ud-dev.conllu.txt'

def logfreq(freq):
    return int(math.log(freq))

def get_byte2ix(training_data):
    byte_to_ix = {'#UNK#':0}
    for sent, _ in training_data:
        for word in sent:
            for c in word:
                for bt in list(c.encode()):
                    if bt not in byte_to_ix:
                        byte_to_ix[bt] = len(byte_to_ix)
    return byte_to_ix 

def get_char2ix(training_data):
    char_to_ix = {'#UNK#':0}
    for sent, _ in training_data:
        for word in sent:
            for char in word:
                if char not in char_to_ix:
                    char_to_ix[char] = len(char_to_ix)
    return char_to_ix

def get_word2ix(training_data):
    word_to_ix = {'#UNK#':0}
    for sent, _ in training_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix

def get_tag2ix(training_data):
    tag_to_ix = {'#UNK#':0}
    for _, tags in training_data:
        for t in tags:
            if t not in tag_to_ix:
                tag_to_ix[t] = len(tag_to_ix)
    return tag_to_ix

def get_tag2logfreq(training_data):
    all_tags = []
    for _, tags in training_data:
        all_tags.extend(tags)
    tag_to_logfreq = Counter(all_tags) 
    # Log of the freqs
    for t in tag_to_logfreq:
        tag_to_logfreq[t] = logfreq(tag_to_logfreq[t])
    return tag_to_logfreq

def read_conllu(filename):
    """
    Read a list of sentences with POS labels from @sefilename. Each
    sentence is a tuple of elements:
    
    BODY      - List of tokens of the sentence.
    POS       - POS label for the sentence.

    """
    data = []

    with open(filename, encoding='utf-8') as sefile:
        token_stack, tag_stack = [], []
        add_to_stack = False
        for line in sefile.readlines()[:100]:
            rand_line = line.split()
            if rand_line == []:
                add_to_stack = False
                data.append((token_stack,tag_stack))
                token_stack, tag_stack = [], []
                continue
            if rand_line[0] == '1':
                add_to_stack = True

            if add_to_stack:
                token_stack.append(rand_line[1])
                tag_stack.append(rand_line[4])
    return data

# training_data = [
#     ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
#     ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
# ]
training_data = read_conllu(TRAIN_FPATH)
test_data = read_conllu(TEST_FPATH)
dev_data = read_conllu(DEV_FPATH)

word_to_ix = get_word2ix(training_data)
byte_to_ix, char_to_ix = get_byte2ix(training_data), get_char2ix(training_data)
tag_to_ix = get_tag2ix(training_data) # {"DET": 0, "NN": 1, "V": 2}
tag_to_logfreq = get_tag2logfreq(training_data) #{"DET": logfreq(3), "NN": logfreq(4), "V": logfreq(2)}

if __name__=="__main__":
    # Check that we don't crash on reading.
    print(tag_to_ix)
    exit(0)

