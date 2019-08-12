import nltk
nltk.download('inaugural')
from nltk.corpus import inaugural, stopwords
from nltk.util import ngrams 
from collections import defaultdict, Counter
import random

def random_vec(dim=100):
    return [random.random() for _ in range(dim)]
 
def get_word_vecs():
    """ word_vecs['target']['context']
    """
    middle_pos = len(n_   grams[0]) // 2



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
