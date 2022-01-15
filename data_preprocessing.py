import numpy as np
import pandas as pd
import collections
from keras.preprocessing.text import text_to_word_sequence

articles = pd.read_csv("all_datasets.csv")
#articles = articles.loc[:, ~articles.columns.str.contains('^Unnamed')]  # remove weird index
articles["text"] = articles["text"].str.lower().apply(text_to_word_sequence)
total_articles = articles.shape[0]  # 44898
#max_len = max(articles["text"].str.len())  # len method works with list, 8375
max_len = max(articles["text"].map(len))  # len method works with list, 8375


test_articles = articles.iloc[:5,:]
total_articles = test_articles.shape[0]  # 44898
max_len = max(test_articles["text"].map(len))  # len method works with list, 8375

# number of articles containing one word from vocab. That is why it's a set
word_freq = collections.Counter(word for art in test_articles["text"] for word in set(art))  # this is better

vocabulary = word_freq.keys()  # 138021, this would be for sentences? nah per article

class TF_IDF:
    def __init__(self):
        print("transforming words to word_freq")

    @staticmethod
    def TF(article, word):  # term_frequency
        #return len([w for w in article if np.any(w == word)])/len(article)  
        return len([w for w in article if w == word])/len(article)  

    @staticmethod
    def IDF(word):  # inverse_document_freq
        return np.log(total_articles/word_freq[word])
        #return np.log(total_articles/word_in_text)
    
    def article_tf_idf(self, article):
        tf_idf_vec = np.zeros((max_len,))  # padding the vector to reach 8375
        art_length = len(article)
        for i, word in enumerate(article):
            tf_idf_vec[i] = self.TF(article, word) * self.IDF(word)
        return tf_idf_vec
    
compute_tf_idf = TF_IDF()


test_articles.loc[:, "text"] = test_articles.apply(lambda row: compute_tf_idf.article_tf_idf(row["text"]), axis=1)


# relevant for decision trees to split data
y_data = np.resize(test_articles["label"].to_numpy(), (total_articles, 1))
x_data = np.resize(test_articles["text"].to_numpy(), (total_articles, max_len))
data = np.concatenate((x_data, y_data), axis=1)
