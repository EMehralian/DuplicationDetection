import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from math import sqrt
from sklearn.metrics import mean_squared_error
import time
from sklearn.linear_model import LogisticRegression
from sklearn import tree, ensemble, linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import math
from scipy import spatial
import xgboost as xgb


LR1 = LogisticRegression(penalty='l1', tol=0.01)
LR2 = LogisticRegression(penalty='l2', tol=0.01)
DT = DecisionTreeClassifier(random_state=0, max_depth=15, min_samples_leaf=2)
RF = RandomForestClassifier(max_depth=10, min_samples_split=2, n_estimators=100, random_state=1,verbose=True)
NN = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(600, 300), random_state=1, activation='relu', verbose=True, max_iter=20)

clf = xgb.XGBClassifier()
metLearn = CalibratedClassifierCV(clf, method='isotonic', cv=2)

leanerSVML1 = LinearSVC(penalty='l1', loss='squared_hinge', dual=False,
                        random_state=0)
leanerSVML2 = LinearSVC(penalty='l2', loss='hinge', dual=True, random_state=0)

clf = svm.SVC(probability=True, verbose=True)


eclf1 = VotingClassifier(estimators=[('lr2', LR2), ('leanerSVML2', leanerSVML2), ('DT', DT)], voting='hard')


kf = KFold(n_splits=10, random_state=None, shuffle=False)


def classifing(data, classifier):

    trainSize = math.ceil(data.shape[0] * 0.8)

    # first 80% for training
    train = data[:trainSize]
    # last 20%  for test
    test = data[trainSize:]

    y_train = train['is_duplicate']
    x_train = train.drop("is_duplicate", axis=1)

    y_test = test['is_duplicate']
    x_test = test.drop("is_duplicate", axis=1)

    X = x_train.values
    y = y_train.values

    # 10-fold validation
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
        classifier.fit(X_train, Y_train)
        print("clf fitted")
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    print("roc", roc_auc_score(y_test, prediction))
    print("precision", precision_score(y_test, prediction))
    print("Fscore", f1_score(y_test, prediction))
    print("accuracy", accuracy_score(y_test, prediction))
    print(recall_score(y_test, prediction, average=None))
    if classifier != leanerSVML1 and classifier != leanerSVML2:
        probPrediction = classifier.predict_proba(x_test)
        print("log loss", log_loss(y_test, probPrediction))
    return prediction


def cosineSimilarity(df):
    prediction = []
    probPrediction = []
    for index, row in df.iterrows():
        result = 1 - spatial.distance.cosine(row[:300], row[300:600])
        probPrediction.append(result)
        if result > 0.7:
            prediction.append(0)
        else:
            prediction.append(1)
    print(probPrediction)
    print("log loss", log_loss(y_test, probPrediction))
    print("Fscore", f1_score(y_test, prediction))
    print("accuracy", accuracy_score(y_test, prediction))
    print(recall_score(y_test, prediction, average=None))


def runClassifer(data):
    temp11 = classifing(data, LR1)
    temp1 = classifing(data, LR2)
    temp = classifing(data, leanerSVML1)
    temp2 = classifing(data, leanerSVML2)
