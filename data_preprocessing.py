import pandas as pd
import nltk
import feather
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
# bag of words does NOT uphold the order of the sentences
# https://stackoverflow.com/questions/55492666/what-is-better-to-use-keras-preprocessing-tokenizer-or-nltk-tokenize  understanding tokenization
#train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

# why tf idf to solve issues with word count https://www.youtube.com/watch?v=76jmgV_ZPUs
# tf idf better at longer documents than word embeddings
