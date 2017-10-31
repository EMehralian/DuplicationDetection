import numpy as np
import pandas as pd
from sklearn import linear_model
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import recall_score
from scipy.spatial import distance
from sklearn.metrics import roc_auc_score

svml1 = linear_model.SGDClassifier(penalty='l1')
svml2 = linear_model.SGDClassifier(penalty='l2')
logisticl1 = linear_model.SGDClassifier(loss='log', penalty='l1')
logisticl2 = linear_model.SGDClassifier(loss='log', penalty='l2')

chunksize = 64


# outer product of question 1 and question2
def outerproduct(df, num_features):
    aggregate = []
    for index, row in df.iterrows():
        outerProduct = np.outer(row[0:num_features], row[num_features:num_features * 2])
        vectorized = outerProduct.ravel()
        aggregate.append(vectorized)
        if (index + 1) % 1000 == 0:
            print(index)
    npArr = np.asarray(aggregate)
    return npArr


# outer product of concatenation of question1 and question2 in itself
def big_outerproduct(df, num_features):
    aggregate = []
    for index, row in df.iterrows():
        outerProduct = np.outer(row[0:num_features * 2], row[0:num_features * 2])
        vectorized = outerProduct.ravel()
        aggregate.append(vectorized)
        if (index + 1) % 1000 == 0:
            print(index)
    npArr = np.asarray(aggregate)
    return npArr


def classify(DataFileName, clf, trainThreshold, testThreshold):
    counter = 0
    for train_df in pd.read_csv(DataFileName, chunksize=chunksize, iterator=True):
        if counter < trainThreshold:
            train_df = train_df.dropna()
            Y = train_df['is_duplicate']
            X = train_df.drop("is_duplicate", axis=1)
            new_x = big_outerproduct(X, 300)
            clf.partial_fit(new_x, Y, classes=np.unique(Y))
            counter += 1
        else:
            print('break')
            break

    print("successfully trained")

    probPrediction = []
    prediction = []
    allY = []
    counter = 0
    for test_df in pd.read_csv(DataFileName, chunksize=chunksize, iterator=True):
        if counter < trainThreshold:
            counter += 1
            continue
        elif trainThreshold <= counter < testThreshold:
            test_df = test_df.dropna()
            for x in test_df['is_duplicate']:
                allY.append(x)
            X = test_df.drop("is_duplicate", axis=1)
            new_x = big_outerproduct(X, 300)
            if clf != svml1 and clf != svml2:
                for x in clf.predict_proba(new_x):
                    probPrediction.append(x)
            temp = clf.predict(new_x)
            for x in temp:
                prediction.append(x)
            counter += 1
        else:
            print("break")
            break

    print("successfully predicted")
    if clf != svml1 and clf != svml2:
        print("log loss", log_loss(allY, probPrediction))
    print("roc", roc_auc_score(allY, prediction))
    print("Fscore", f1_score(allY, prediction))
    print("accuracy", accuracy_score(allY, prediction))
    print(recall_score(allY, prediction, average=None))
    print('clf', clf)


def partialClassifier():
    # find the name of the file contains question features from questionVectors.py file and replace it here
    DataFileName = 'sum.csv'
    classify(DataFileName, logisticl1, 994, 1242)
    classify(DataFileName, logisticl2, 994, 1242)
    classify(DataFileName, svml1, 994, 1242)
    classify(DataFileName, svml2, 994, 1242)
