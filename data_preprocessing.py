import numpy as np
import pandas as pd
import collections

X_train = pd.read_feather("X_train.feather")
X_test = pd.read_feather("X_test.feather")
X_valid = pd.read_feather("X_valid.feather")
y_train = pd.read_feather("y_train.feather")
y_test = pd.read_feather("y_test.feather")
y_valid = pd.read_feather("y_valid.feather")
articles = X_train.values.tolist()
# number of articles containing one word from vocab. That is why it's a set
frequencies = collections.Counter(word for art in articles for word in set(art[0]))  # this is better
vocabulary = frequencies.keys()
word2index = {word: e for e, word in enumerate(vocabulary)}

total_articles = X_train.shape[0]


def TF(art, word):  # term_frequency
    #return len([w for w in art if w == word])/len(art)  # takes too long
    return len([w for w in art if np.any(w[0] == word)])/len(art)  # takes too long

def IDF(word, frequencies, total_articles):  # inverse_document_freq
    if frequencies[word]:
        counter = frequencies[word] + 1
    else:
        counter = 1
    return np.log(total_articles/counter)

def TF_IDF(article):
    tf_idf_vec = np.zeros((len(vocabulary),))
    for word in article[0]:
        tf_idf_vec[word2index[word]] = TF(article, word) * IDF(word, frequencies, total_articles)
    return tf_idf_vec


tf_idf_vectors = [TF_IDF(art) for art in articles]
#stored_vectors = pd.DataFrame(np.concatenate(tf_idf_vectors))  # dangerous 4 this task
stored_vectors = pd.DataFrame(np.row_stack(tf_idf_vectors))


# https://www.askpython.com/python/examples/tf-idf-model-from-scratch
# https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089
# https://www.youtube.com/watch?v=vZAXpvHhQow
# https://www.youtube.com/watch?v=lBO1L8pgR9s


# bag of words does NOT uphold the order of the sentences
# https://stackoverflow.com/questions/55492666/what-is-better-to-use-keras-preprocessing-tokenizer-or-nltk-tokenize  understanding tokenization
#train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

# why tf idf to solve issues with word count https://www.youtube.com/watch?v=76jmgV_ZPUs
# tf idf better at longer documents than word embeddings
# https://towardsdatascience.com/understanding-word-embeddings-with-tf-idf-and-glove-8acb63892032
# tf-idf is a sparse vector representation instead of a dense one like GloVe. GloVe is closer to a Word2vec, i.e. a word embedding

# concat vs stack https://deeplizard.com/learn/video/kF2AlpykJGY
