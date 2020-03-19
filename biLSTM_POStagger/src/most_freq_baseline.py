import numpy as np
from collections import Counter
from data_ud import word_to_ix,byte_to_ix,char_to_ix,tag_to_ix,freq_to_ix,token_to_freq,\
                    training_data,dev_data,test_data

def get_mostfreq_tag(training_data):
    """Should not be used"""
    all_tags = []
    for _, tags in training_data:
        all_tags.extend(tags)
    tag_to_freq = Counter(all_tags) 
    return tag_to_freq.most_common()[0][0]

MOST_FREQ_TAG = get_mostfreq_tag(training_data)

def get_vanilla_acc(data):
    correct, total = 0, 0
    for _, tags in data:
        total += len(tags)
        for t in tags:
            if t==MOST_FREQ_TAG:
                correct += 1
    return correct/total 

if __name__ == "__main__":
    test_acc = get_vanilla_acc(test_data)
    print('test acc: %.9f%%' % (test_acc))