import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from keras.models import Sequential
from keras.layers import Dense

seed = 7
np.random.seed(seed)

svml1 = linear_model.SGDClassifier(penalty='l1')
svml2 = linear_model.SGDClassifier(penalty='l2')
logisticl1 = linear_model.SGDClassifier(loss='log', penalty='l1')
logisticl2 = linear_model.SGDClassifier(loss='log', penalty='l2')
MLPclf = MLPClassifier(activation='relu', learning_rate='constant',
 alpha=1e-4, hidden_layer_sizes=(80,40), random_state=1, batch_size=1,verbose= False,
 max_iter=20, warm_start=True)

NN40 = MLPClassifier( alpha=1e-4, hidden_layer_sizes=(40,), random_state=1, activation='relu', verbose=False, max_iter=20)
NN1600 = MLPClassifier( alpha=1e-4, hidden_layer_sizes=(1600,), random_state=1, activation='relu', verbose=False, max_iter=20)

# create model
model = Sequential()
model.add(Dense(600, input_dim=600, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


print("big_outerproduct")

chunksize = 64


def averaging_outerproduct(df, num_features):
    aggregate = []
    for index, row in df.iterrows():
        outerProduct = np.outer(row[0:num_features], row[num_features:num_features * 2])
        vectorized = outerProduct.ravel()
        aggregate.append(vectorized)
        if (index + 1) % 1000 == 0:
            print(index)
    npArr = np.asarray(aggregate)
    return npArr


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


def classify(clf, trainThreshold, testThreshold):
    counter = 0
    for train_df in pd.read_csv('QuConcatR40.csv', chunksize=chunksize, iterator=True):
        # print(counter)
        if counter < trainThreshold:
            train_df = train_df.drop('Unnamed: 0', 1)
            train_df = train_df.dropna()
            # print(train_df.shape)
            Y = train_df['is_duplicate']
            X = train_df.drop("is_duplicate", axis=1)
            new_x = big_outerproduct(X, 40)
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
    for test_df in pd.read_csv('QuConcatR40.csv', chunksize=chunksize, iterator=True):
        if counter < trainThreshold:
            counter += 1
            continue
        if counter < testThreshold:
            test_df = test_df.drop('Unnamed: 0', 1)
            test_df = test_df.dropna()
            for x in test_df['is_duplicate']:
                allY.append(x)
            X = test_df.drop("is_duplicate", axis=1)
            new_x = big_outerproduct(X, 40)
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


classify(NN40, 5049, 6312)
