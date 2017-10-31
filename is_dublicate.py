# create model
import math
import pandas as pd

from siamese import *
from keras.optimizers import RMSprop, SGD, Adam
from sklearn.metrics import accuracy_score

data = pd.read_csv('base.csv')
print(data.head())
data = data.drop('Unnamed: 0', 1)
print(data.head())
print("------------------------------------------------")
print(data[100:110])

trainSize = int(data.shape[0]*0.8)
testSize = data.shape[0] - trainSize
print(trainSize)
train = data[:trainSize]
test = data[trainSize:]

y_train = train['is_duplicate']
x_train = train.drop("is_duplicate", axis=1)

y_test = test['is_duplicate']
x_test = test.drop("is_duplicate", axis=1)
net = create_network(300)

# train
# optimizer = SGD(lr=1, momentum=0.8, nesterov=True, decay=0.004)
optimizer = Adam(lr=0.0001)
net.compile(loss='hinge', optimizer=optimizer)

attr1Train = []
attr2Train = []
for index, row in x_train.iterrows():
    attr1Train.append(row[:300])
    attr2Train.append(row[300:])
attr1Train = np.asarray(attr1Train)
attr2Train = np.asarray(attr2Train)
attr1Test = []
attr2Test = []
for index, row in x_test.iterrows():
    attr1Test.append(row[:300])
    attr2Test.append(row[300:])
attr1Test = np.asarray(attr1Test)
attr2Test = np.asarray(attr2Test)

X_train = np.zeros([trainSize, 2, 300])
X_test = np.zeros([testSize, 2, 300])
Y_train = np.zeros([trainSize])
Y_test = np.zeros([testSize])

X_train[:, 0, :] = attr1Train
X_train[:, 1, :] = attr2Train
Y_train = y_train.values

X_test[:, 0, :] = attr1Test
X_test[:, 1, :] = attr2Test
Y_test = y_test.values

for epoch in range(50):
    net.fit([X_train[:, 0, :], X_train[:, 1, :]], Y_train,
            validation_data=([X_test[:, 0, :], X_test[:, 1, :]], Y_test),
            batch_size=128, nb_epoch=1, shuffle=True)

    # compute final accuracy on training and test sets
    pred = net.predict([X_test[:, 0, :], X_test[:, 1, :]])
    te_acc = accuracy_score(pred.ravel() < 0.5, Y_test)

    #    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    # print("accuracy", accuracy_score(y_test, prediction))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
