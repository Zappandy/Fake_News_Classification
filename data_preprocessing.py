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
max_len = max(articles["text"].str.len())  # len method works with list, 8375

# number of articles containing one word from vocab. That is why it's a set

frequencies = collections.Counter(word for art in articles["text"] for word in set(art))  # this is better
#vocabulary = frequencies.keys()  # 138021, this would be for sentences? nah per article
print(max_len)
raise SystemExit


class TF_IDF:
    def __init__(self):
        print("transforming words to frequencies")

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
        tf_idf_vec = np.zeros((max_len,))  # padding the vector 
        for word in article:
            tf_idf_vec[word2index[word]] = self.TF(article, word) * self.IDF(word)
        return tf_idf_vec
    


example = TF_IDF()
example = example.article_tf_idf(articles["text"].iloc[0,])   # no need to .flatten(). Already 1D
print(example.size)

