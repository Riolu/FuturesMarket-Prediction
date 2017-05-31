import numpy as np
from sklearn.naive_bayes import GaussianNB
import time

X = np.load("data/data_X.npy")
y = np.load("data/data_y.npy")

index = int(len(y)*0.7)

X_train = X[:index]
y_train = y[:index]

X_test = X[index:len(y)]
y_test = y[index:len(y)]


def nb_classify(X_train, y_train, X_test, y_test):
    t0 = time.time()
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    print("nb done in %0.3fs" % (time.time() - t0))
    print (1 - np.sum(np.abs(clf.predict(X_test) - y_test)) / float(0.3 * len(y)))

nb_classify(X_train, y_train, X_test, y_test)