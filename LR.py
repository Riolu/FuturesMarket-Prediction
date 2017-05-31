import numpy as np
from sklearn.linear_model import LinearRegression
import time

X = np.load("data/data_X.npy")
y = np.load("data/data_y.npy")

index = int(len(y)*0.7)

X_train = X[:index]
y_train = y[:index]

X_test = X[index:len(y)]
y_test = y[index:len(y)]


def lr_classify(X_train, y_train, X_test, y_test):
    t0 = time.time()
    clf = LinearRegression()
    clf.fit(X_train,y_train)
    print("lr done in %0.3fs" % (time.time() - t0))
    print (1 - np.sum(np.abs(clf.predict(X_test) - y_test)) / float(0.3 * len(y)))

lr_classify(X_train, y_train, X_test, y_test)