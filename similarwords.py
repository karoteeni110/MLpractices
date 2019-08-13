import nltk
nltk.download('inaugural')
nltk.download('wordnet')
from nltk.corpus import inaugural, stopwords
from nltk.util import ngrams 
from collections import defaultdict, Counter
from random import shuffle, choice
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from nltk.stem import WordNetLemmatizer

DIM = 100

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
    
def get_word_vecs():
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

def mostsimilar(word, most = 10):
    pairwise_sims = cosine_similarity(vec_dict[word].reshape(1,-1), list(vec_dict.values()))
    # print(pairwise_sims)
    maxargs = pairwise_sims[0].argsort()[most*(-1):][::-1]
    print('Most %d similar words for `%s`:' % (most, word))
    print()
    print('Word', '          ', 'Cosine Similarity')
    for arg in maxargs:
        compared_word = list(vec_dict.keys())[arg] 
        if compared_word != word:
            print(compared_word, ' ' * (14-len(compared_word)) , pairwise_sims[0][arg])
    print()

def random_similarwords(examples=10):
    for _ in range(examples):
        word = choice(list(vec_dict.keys()))
        mostsimilar(word)

if __name__ == "__main__":
    # Get vocabulary & ngrams
    window_size = 5
    stopwords = stopwords.words('english')
    lmtzer = WordNetLemmatizer()
    filtered_sents = [tuple(lmtzer.lemmatize(word.lower()) \
                    for word in sent if word.isalnum() and not word in stopwords) \
                    for sent in inaugural.sents()]
    n_grams = get_ngrams(filtered_sents)
    vocab = set(word for sent in filtered_sents for word in sent)
    vec_dict = get_word_vecs()

    mostsimilar('demoralizes')
    print(for i in vec_dict['demoralizes'])
    for sent in filtered_sents:
        if 'demoralizes' in sent:
            print(sent)
    # random_similarwords(examples=10)


