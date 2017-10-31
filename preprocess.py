import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import PorterStemmer


train = pd.read_csv('Data/train.csv').fillna(" ")
print("data has been loaded")

# train = train.drop(105780)
# train = train.drop(201841)
train = train.dropna()
print("dropped")

x_train = train

print("choosed")

y_train = x_train["is_duplicate"]
x_train = x_train.drop("is_duplicate", axis=1)

# change data type from string to object
x_train = x_train.astype('object')

print("changed")
tokenizer = nltk.RegexpTokenizer(r'\w+')

# pre processing function which get an sentence and returns list of its words
# 1. just keeps alphabetical letters
# 2. tokenize each sentence
# 3. not remove stopwords
# 4. change to everything lower case
# swords = stopwords.words('english')

def text_to_wordlist(text):
    # Clean the text, with the option to remove stop_words and to stem words.
    ps = PorterStemmer()
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"([0-9])[Kk] ", r"\1000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text)
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)
    text = re.sub(r"demonitization", "demonetization", text)
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text)
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"0rs ", " rs ", text)
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text)
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text)
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)

    tokenized = tokenizer.tokenize(text)
    wordsList = [word.lower() for word in tokenized]
    # stemList = [ps.stem(w) for w in wordsList]
    if len(wordsList) < 5:
        return None
    return " ".join(wordsList)


# deploy pre-process in the train data and store result(words of every sentence) in the dataframe
tockenized1 = []
tockenized2 = []
# deploy pre-process in the train data and store result(words of every sentence) in the dataframe
for index, row in x_train.iterrows():
    print(index)
    temp1 = text_to_wordlist(row["question1"])
    temp2 = text_to_wordlist(row["question2"])
    tockenized1.append(temp1)
    tockenized2.append(temp2)

tockenized1_array = np.asarray(tockenized1)
tockenized2_array = np.asarray(tockenized2)
x_train["question1"] = pd.Series(tockenized1_array, index=x_train.index)
x_train["question2"] = pd.Series(tockenized2_array, index=x_train.index)
x_train = x_train.dropna()

x_train = pd.concat([x_train, y_train], axis=1)

print("preprocess complieted")

x_train.to_csv('processed_train.csv', index=True)
