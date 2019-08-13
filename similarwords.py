import nltk
nltk.download('inaugural')
from nltk.corpus import inaugural, stopwords
from nltk.util import ngrams 
from collections import defaultdict, Counter
from random import shuffle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

DIM = 10

def random_vec(dim=DIM):
    vec = np.zeros(DIM, dtype=int)
    vec[0:5] = 1
    shuffle(vec)
    return vec

def get_ngrams(filtered_sents):
    n_grams = []
    for sent in filtered_sents:
        n_grams.extend(list(ngrams(sent, window_size)))
    return n_grams

def normalise(vec_dict):
    for w in vec_dict:
        vec_dict[w] = vec_dict[w] / sum(vec_dict.values())
    return vec_dict
    
def get_word_vecs(filtered_sents):
    n_grams = get_ngrams(filtered_sents)
    vocab = set(word for sent in filtered_sents for word in sent)

    index_vec, context_vec = {}, {}
    for w in vocab:
        index_vec[w] = random_vec()
        context_vec[w] = np.zeros(DIM, dtype=int)
    
    middle_pos = len(n_grams[0]) // 2
    for ng in n_grams:
        focus = ng[middle_pos]
        context = ng[:middle_pos] + ng[middle_pos+1:]
        for w in context:
            context_vec[focus] += index_vec[w]
    return normalise(context_vec)

def word2idx(word):
    return word2idx_dict[word]

def mostsimilar(word, vec_dict, most = 3):
    pairwise_sims = cosine_similarity(vec_dict[word].reshape(1,-1), list(vec_dict.values()))
    # print(pairwise_sims)
    maxargs = pairwise_sims[0].argsort()[most*(-1):][::-1]
    print('Most %d similar words for `%s`:' % (most, word))
    print('Word', '          ', 'Cosine Similarity')
    for arg in maxargs:
        compared_word = list(vec_dict.keys())[arg] 
        print(compared_word, ' ' * (14-len(compared_word)) , pairwise_sims[0][arg])
    print()


if __name__ == "__main__":
    # Get vocabulary & ngrams
    window_size = 5
    stopwords = stopwords.words('english')
    filtered_sents = [tuple(word.lower() for word in sent if word.isalnum() and not word in stopwords) \
                        for sent in inaugural.sents()[:10]]
    vec_dict = get_word_vecs(filtered_sents)
    word2idx_dict = {word:i for i, word in enumerate(vec_dict.keys())}
    # a,b = vec_dict['good'], vec_dict['great']
    # c = vec_dict['bad']
    mostsimilar('good', vec_dict)
    # print('Cosine distance between `good` and `great`:', cosine_similarity(a,b))
    # print('Cosine distance between `good` and `bad`:', cosine_similarity(a,c))
