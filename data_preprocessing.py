import pandas as pd
import nltk
#nltk.download('punkt') only download once
from nltk.tokenize import word_tokenize # if possible, otherwise .split()

fake = pd.read_csv("Fake.csv",usecols=["text"]) # extracting only article texts
true = pd.read_csv("True.csv",usecols=["text"])

true["label"] = 0 
fake["label"] = 1

df = pd.concat([true, fake]).sample(frac=1) # shuffle rows to create a representative train/test set
df["text"] = df["text"].str.lower().apply(word_tokenize)
print(df.head())
