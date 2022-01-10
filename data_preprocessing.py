import numpy as np
import pandas as pd
import collections
from keras.preprocessing.text import text_to_word_sequence

articles = pd.read_csv("all_datasets.csv")
articles["text"] = articles["text"].str.lower().apply(text_to_word_sequence)
total_articles = articles.shape[0]  # 44898
max_len = max(articles["text"].str.len())  # len method works with list

# number of articles containing one word from vocab. That is why it's a set
frequencies = collections.Counter(word for art in articles["text"] for word in set(art))  # this is better
vocabulary = frequencies.keys()  # 138021, this would be for sentences? nah per article
word2index = {word: e for e, word in enumerate(vocabulary)}

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
        tf_idf_vec = np.zeros((max_len,))  # padding the vector to reach 8375
        for word in article:
            tf_idf_vec[word2index[word]] = self.TF(article, word) * self.IDF(word)
        return tf_idf_vec
    
compute_tf_idf = TF_IDF()
example = compute_tf_idf.article_tf_idf(articles["text"].iloc[0,]) 
