import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from nltk import word_tokenize
from nltk.corpus import stopwords


data = pd.read_csv('allQuestions.csv')

data = data.drop(['id', 'qid1', 'qid2'], axis=1)


TAG_RE = re.compile(r'<code>[^>]+</code>')
TAG_REAll = re.compile(r'<[^>]+>')
stop_words = stopwords.words('english')


def remove_tags(text):
    text = TAG_RE.sub('', text)
    return TAG_REAll.sub('', text)


def preprocess(s):
    try:
        s = remove_tags(s)
    except:
        print(s)
        print("something goes wrong!")
    words = str(s).lower()#.decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    sentece = ""
    for word in words:
        sentece += " " + word
    return sentece

# q1FVecs = np.empty((data.shape[0],1))
# q2FVecs = np.empty((data.shape[0],1))
q1FVecs = []
q2FVecs = []

for i, q in tqdm(enumerate(data.question1.values)):
    q1FVecs.append(preprocess(q))
print("question1 calculated")

for i, q in tqdm(enumerate(data.question2.values)):
    q2FVecs.append(preprocess(q))
print("question2 calculated")


dfFeatures = np.vstack((q1FVecs, q2FVecs)).T

train_df = pd.DataFrame(dfFeatures)


train_df1 = pd.concat([train_df, data["isDuplicate"]], axis=1)
train_df2 = pd.DataFrame(train_df1)
print(train_df2.shape)
print(train_df2.head())

train_df2.to_csv('processedAU.csv', index=True)
