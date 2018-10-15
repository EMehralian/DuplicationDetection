import gensim
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.decomposition import PCA, IncrementalPCA


# find a numerical representation for each question using
# google news pre trained word2vec model for each word in the
# question and then averaging the feature vectors of words
# of a question as a feature vector of the whole sentence


def loadTwitterModel():
    with open('glove/glove.twitter.27B.25d.txt', 'r') as f:
        model = {}
        for line in f:
            vals = line.rstrip().split(' ')
            model[vals[0]] = [float(x) for x in vals[1:]]
    return model


stop_words = stopwords.words('english')
data = pd.read_csv('processed_train.csv')

data = data.drop(['qid1', 'qid2'], axis=1)


def SumSent2vec(s, model, num_features):
    words = str(s).lower()  # .decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


def AvgSent2vec(s, model, num_features):
    words = str(s).lower()  # .decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    wordNum = 0
    global counter
    for w in words:
        try:
            wordNum += 1
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.mean(axis=0)
    return v / np.sqrt((v ** 2).sum())


def pairWiseProduct(s, model, num_features):
    words = str(s).lower()  # .decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = [1] * num_features
    for word in M:
        for i in range(num_features):
            v[i] = v[i] * word[i]

    v = np.array(v)

    return v / np.sqrt((v ** 2).sum())


def difff(q1, q2, num_features):
    return np.subtract(q1, q2)


def absdifff(q1, q2, num_features):
    difference = np.subtract(q1, q2)
    return np.absolute(difference)


def summ(q1, q2, num_features):
    return np.add(q1, q2)


def multt(q1, q2, num_features):
    return np.multiply(q1, q2)


q1FVecs = np.zeros((data.shape[0], 300))
q2FVecs = np.zeros((data.shape[0], 300))


def execute(embModel, aggMethod, reduced):
    model = 0
    if embModel == "googleNews":
        model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        print("googleLoaded")
    elif embModel == "glove":
        model = loadTwitterModel()
    else:
        print("specified embegging model is not defined")

    if aggMethod == "sum":
        for i, q in tqdm(enumerate(data.question1.values)):
            q1FVecs[i, :] = SumSent2vec(q, model, 300)
        print("question1 calculated")

        for i, q in tqdm(enumerate(data.question2.values)):
            q2FVecs[i, :] = SumSent2vec(q, model, 300)
        print("question2 calculated")

    elif aggMethod == "Average":
        print("average")
        for i, q in tqdm(enumerate(data.question1.values)):
            q1FVecs[i, :] = AvgSent2vec(q, model, 300)
        print("question1 calculated")

        for i, q in tqdm(enumerate(data.question2.values)):
            q2FVecs[i, :] = AvgSent2vec(q, model, 300)
        print("question2 calculated")

    elif aggMethod == "pairWiseProduct":
        for i, q in tqdm(enumerate(data.question1.values)):
            q1FVecs[i, :] = pairWiseProduct(q, model, 300)
        print("question1 calculated")

        for i, q in tqdm(enumerate(data.question2.values)):
            q2FVecs[i, :] = pairWiseProduct(q, model, 300)
        print("question2 calculated")

    dfFeatures = np.concatenate((q1FVecs, q2FVecs), axis=1)

    difFeatures = np.zeros((data.shape[0], 25))
    for i, q in tqdm(enumerate(dfFeatures)):
        difFeatures[i, :] = absdifff(q[:25], q[25:], 25)

    mulFeatures = np.zeros((data.shape[0], 25))
    for i, q in tqdm(enumerate(dfFeatures)):
        mulFeatures[i, :] = multt(q[:25], q[25:], 25)

    absmul = np.concatenate((difFeatures, mulFeatures), axis=1)

    train_df = pd.DataFrame(dfFeatures)

    train_df1 = pd.concat([train_df, data["is_duplicate"]], axis=1)

    train_df2 = pd.DataFrame(train_df1)

    if reduced:
        reducedDF = dimentionReduction(train_df2.dropna())
        reducedDF.to_csv('QuAbsMulR40.csv', index=True)
        return reducedDF
    print('here')

    difFeatures = np.zeros((data.shape[0], 300))
    for i, q in tqdm(enumerate(dfFeatures)):
        difFeatures[i, :] = absdifff(q[:300], q[300:], 300)
    #
    mulFeatures = np.zeros((data.shape[0], 300))
    for i, q in tqdm(enumerate(dfFeatures)):
        mulFeatures[i, :] = multt(q[:300], q[300:], 300)
    #
    # sumFeatures = np.zeros((data.shape[0], 300))
    # for i, q in tqdm(enumerate(dfFeatures)):
    #      sumFeatures[i, :] = summ(q[:300], q[300:], 300)

    absmul = np.concatenate((difFeatures, mulFeatures), axis=1)

    train_df = pd.DataFrame(absmul)

    train_df1 = pd.concat([train_df, data["is_duplicate"]], axis=1)

    train_df2 = pd.DataFrame(train_df1)

    train_df2.to_csv('W2Vabsmul.csv', index=True)
    return train_df


def dimentionReduction(local_df):
    print(local_df.shape)
    Q1 = local_df.iloc[:, :300].values
    Q2 = local_df.iloc[:, 300:600].values

    print(Q1.shape)
    print(Q2.shape)
    allQs = np.concatenate([Q1, Q2])
    print(allQs.shape)
    ipca = IncrementalPCA(n_components=40, batch_size=512)
    reducedQ1 = ipca.fit_transform(allQs)
    print(ipca.explained_variance_ratio_.cumsum())
    print(reducedQ1.shape)
    qsize = int(reducedQ1.shape[0] / 2)
    print(qsize)

    q1 = reducedQ1[:qsize]
    q2 = reducedQ1[qsize:]
    dfFeaturesNew2 = np.concatenate((q1, q2), axis=1)

    difFeatures = np.zeros((data.shape[0], 40))
    for i, q in tqdm(enumerate(dfFeaturesNew2)):
        difFeatures[i, :] = absdifff(q[:40], q[40:], 40)

    # sumFeatures = np.zeros((data.shape[0], 40))
    # for i, q in tqdm(enumerate(dfFeaturesNew2)):
    #      sumFeatures[i, :] = summ(q[:40], q[40:], 40)

    mulFeatures = np.zeros((data.shape[0], 40))
    for i, q in tqdm(enumerate(dfFeaturesNew2)):
        mulFeatures[i, :] = multt(q[:40], q[40:], 40)
    #
    absmul = np.concatenate((difFeatures, mulFeatures), axis=1)

    train_df1 = pd.DataFrame(absmul)
    train_df2 = pd.concat([train_df1.reset_index(drop=True), local_df["is_duplicate"].reset_index(drop=True)], axis=1)
    return train_df2


a = execute("googleNews", "Average", True)  # True if you want reduced dimension representation
print(a.dropna().shape)
