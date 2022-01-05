import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
#nltk.download('punkt') only download once
from nltk.tokenize import word_tokenize # if possible, otherwise .split()

fake = pd.read_csv("Fake.csv",usecols=["text"]) # extracting only article texts
true = pd.read_csv("True.csv",usecols=["text"])

true["label"] = 0 
fake["label"] = 1

df = pd.concat([true, fake]).sample(frac=1) # shuffle rows to create a representative train/test set
df["text"] = df["text"].str.lower().apply(word_tokenize)
print(df.head())
print(df.shape)
#X_train, X_test, y_train, y_set = train_test_split(df["text"], df["label"], test_size=0.33)
X_train, X_test, y_train, y_set = train_test_split(df["text"], df["label"], train_size=0.8, test_size=0.2)
X_train, X_valid, y_train, y_valid = train_test_split(df["text"], df["label"], train_size=0.9, test_size=0.1)
#print(type(X_train))
print(X_train.head())

#train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
