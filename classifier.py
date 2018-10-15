import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import math
import xgboost as xgb


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


print("10000000000")
data = pd.read_csv("ReducedConcatinate40.csv").dropna()
#
print("data", data.shape)
print("head", data.head())
data = data.drop('Unnamed: 0', 1)
data = data.sample(frac=1)
data = data.sample(frac=1).reset_index(drop=True)
print("data", data.shape)
print("head", data.head())

trainSize = math.ceil(data.shape[0] * 0.8)

train = data[:trainSize]
test = data[trainSize:]
print('Train_labels', train['is_duplicate'].value_counts())
print('Test_labels', test['is_duplicate'].value_counts())

y_train = train["is_duplicate"]
x_train = train.drop("is_duplicate", axis=1)


y_test = test["is_duplicate"]
x_test = test.drop("is_duplicate", axis=1)


LR1 = LogisticRegression(penalty='l1', tol=0.01)
LR2 = LogisticRegression(penalty='l2', tol=0.01)
DT = DecisionTreeClassifier(random_state=0, max_depth=15, min_samples_leaf=2)
RF = RandomForestClassifier(max_depth=10, min_samples_split=2, n_estimators=100, random_state=1, verbose=True)
NN40 = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(40,), random_state=1, activation='relu',
                   verbose=True, max_iter=20)

NN1600 = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(1600,), random_state=1, activation='relu',
                   verbose=True, max_iter=20)

MLPclf = MLPClassifier(activation='relu', learning_rate='constant',
                       alpha=1e-4, hidden_layer_sizes=(80, 40), random_state=1, batch_size=1, verbose=False,
                       max_iter=20, warm_start=True)

clf = xgb.XGBClassifier()
metLearn = CalibratedClassifierCV(clf, method='isotonic', cv=2)

leanerSVML1 = LinearSVC(penalty='l1', loss='squared_hinge', dual=False,
                        random_state=0)
leanerSVML2 = LinearSVC(penalty='l2', loss='hinge', dual=True, random_state=0)

clf = svm.SVC(probability=True, verbose=True)

eclf1 = VotingClassifier(estimators=[('lr2', LR2), ('leanerSVML2', leanerSVML2), ('DT', DT)], voting='hard')

kf = KFold(n_splits=10, random_state=None, shuffle=False)


X = x_train.values
y = y_train.values


def classifing(classifier):
    classifier.fit(x_train, y_train)
    print("fitted")
    prediction = classifier.predict(x_test)

    cm = confusion_matrix(y_test, prediction)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("conf", cm)

    print("roc", roc_auc_score(y_test, prediction))
    # print("precision", precision_score(y_test, prediction))
    print("Fscore", f1_score(y_test, prediction))
    print("accuracy", accuracy_score(y_test, prediction))
    print(recall_score(y_test, prediction, average=None))
    if classifier != leanerSVML1 and classifier != leanerSVML2:
        probPrediction = classifier.predict_proba(x_test)
        print("log loss", log_loss(y_test, probPrediction))
    return prediction


print("its here")

temp1 = classifing(LR2)
#
temp2 = classifing(leanerSVML2)
#
temp3 = classifing(NN40)

temp4 = classifing(NN1600)
