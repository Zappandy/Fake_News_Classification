import numpy as np
import pandas as pd
import collections
from keras.preprocessing.text import text_to_word_sequence

# probably bad
#X_train = pd.read_feather("X_train.feather")
#X_test = pd.read_feather("X_test.feather")
#X_valid = pd.read_feather("X_valid.feather")
#y_train = pd.read_feather("y_train.feather")
#y_test = pd.read_feather("y_test.feather")
#y_valid = pd.read_feather("y_valid.feather")

articles = pd.read_csv("all_datasets.csv")
articles["text"] = articles["text"].str.lower().apply(text_to_word_sequence)
total_articles = articles.shape[0]
# number of articles containing one word from vocab. That is why it's a set

frequencies = collections.Counter(word for art in articles["text"] for word in set(art))  # this is better
vocabulary = frequencies.keys()




class TF_IDF:
    def __init__(self):
        print("transforming words to frequencies")
        # store the length of each vector, shorter vectors should be padded with 0s at the end
        

    @staticmethod
    def TF(art, word):  # term_frequency
        return len([w for w in art if np.any(w[0] == word)])/len(art)  

    @staticmethod
    def IDF(word):  # inverse_document_freq
        if frequencies[word]:
            counter = frequencies[word] + 1
        else:
            counter = 1
        return np.log(total_articles/counter)
    
    def article_tf_idf(self, article):
        tf_idf_vec = np.zeros((len(vocabulary),))
        for word in article:
            tf_idf_vec[word2index[word]] = self.TF(article, word) * self.IDF(word)
        return tf_idf_vec
    
example = TF_IDF()
example = example.article_tf_idf(articles["text"].iloc[0,])   # no need to .flatten(). Already 1D
print(example.size)
raise SystemExit

# IGNORE THINGS UNDER THIS LINE
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
