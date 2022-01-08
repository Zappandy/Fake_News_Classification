import pandas as pd
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
# better train, test, split methods 
# https://stackoverflow.com/questions/45221940/creating-train-test-val-split-with-stratifiedkfold
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], train_size=0.8, test_size=0.2)  # test_size=0.33?
X_train, X_valid, y_train, y_valid = train_test_split(df["text"], df["label"], train_size=0.9, test_size=0.1)
datasets = [X_test, y_test, X_train, y_train, X_valid, y_valid]
datasets = [data.to_frame() for data in datasets]
datasets[0].reset_index(drop=True, inplace=True)
datasets[1].reset_index(drop=True, inplace=True)
datasets[2].reset_index(drop=True, inplace=True)
datasets[3].reset_index(drop=True, inplace=True)
datasets[4].reset_index(drop=True, inplace=True)
datasets[5].reset_index(drop=True, inplace=True)

datasets[0].to_feather("X_test.feather")
datasets[1].to_feather("y_test.feather")
datasets[2].to_feather("X_train.feather")
datasets[3].to_feather("y_train.feather")
datasets[4].to_feather("X_valid.feather")
datasets[5].to_feather("y_valid.feather")
