import numpy as np
import pandas as pd
import nltk
import feather
import collections
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import text_to_word_sequence

# prioritizing keras over nltk for preprocessing
#nltk.download('punkt') only download once
from nltk.tokenize import word_tokenize # if possible, otherwise .split()

# convert to feather for faster processing and less space usage
#fake = pd.read_csv("Fake.csv",usecols=["text"]) # extracting only article texts
#true = pd.read_csv("True.csv",usecols=["text"])
#fake.to_feather("Fake_News.feather")
#true.to_feather("Factual_News.feather")

fake = pd.read_feather("Fake_News.feather")
true = pd.read_feather("Factual_News.feather")

true["label"] = 0 
fake["label"] = 1

df = pd.concat([true, fake]).sample(frac=1) # shuffle rows to create a representative train/test set
# since they are articles, maybe sent tokenize is better
df["text"] = df["text"].str.lower().apply(text_to_word_sequence)  # 23481 rows, takes less than a min
#X_train, X_test, y_train, y_set = train_test_split(df["text"], df["label"], test_size=0.33)
X_train, X_test, y_train, y_set = train_test_split(df["text"], df["label"], train_size=0.8, test_size=0.2)
X_train, X_valid, y_train, y_valid = train_test_split(df["text"], df["label"], train_size=0.9, test_size=0.1)
#X_train.reset_index(inplace=True) do we need index? no, this is a series anyway
# y_train.to_frame()  # does not have in place?
articles = X_train.values.tolist()
# number of articles containing one word from vocab. That is why it's a set
frequencies = collections.Counter(word for art in articles for word in set(art))  # this is better
vocabulary = frequencies.keys()
word2index = {word: e for e, word in enumerate(vocabulary)}

total_articles = X_train.shape[0]

def TF(art, word):  # term_frequency
    return len([w for w in art if w == word])/len(art)

def IDF(word):  # inverse_document_freq
    if frequencies[word]:
        counter = frequencies[word] + 1
    else:
        counter = 1
    return np.log(total_articles/counter)

def TF_IDF(article):
    tf_idf_vec = np.zeros((len(vocabulary),))
    for word in article:
        tf_idf_vec[word2index[word]] = TF(article, word) * IDF(word)
    return tf_idf_vec

tf_idf_vectors = [tf_idf_vectors(art) for art in articles]

print(tf_idf_vectors[0])
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

