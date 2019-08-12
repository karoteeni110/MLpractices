import nltk
nltk.download('inaugural')
from nltk.corpus import inaugural, stopwords
from nltk.util import ngrams 
from collections import defaultdict, Counter
from random import shuffle

def random_vec(dim=100):
    vec = [1.0] * 5 + [0.0] * (dim-5)
    shuffle(vec)
    return vec
 
def get_word_vecs():
    """ word_vecs[]
    """
    initial_vecs = defaultdict()
    for word in vocab:
        initial_vecs[word] = random_vec()
        
    middle_pos = len(n_grams[0]) // 2
    word_vecs = Counter(int)
    for window in n_grams:
        target = window[middle_pos]


if __name__ == "__main__":
    # Get vocabulary & ngrams
    window_size = 5
    stopwords = stopwords.words('english')
    filtered_sents = [tuple(word.lower() for word in sent if word.isalnum() and not word in stopwords) \
                        for sent in inaugural.sents()]
    vocab = set(word for sent in filtered_sents for word in sent)
    
    n_grams = []
    for sent in filtered_sents:
        n_grams.extend(list(ngrams(sent, window_size)))
    
    print(n_grams[:5])
