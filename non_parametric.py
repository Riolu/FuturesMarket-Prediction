import pickle
import math
import numpy as np
f = open("data0v2.pkl","rb")
x_train= pickle.load(f)
f.close()
f = open("label0v2.pkl","rb")
total = 1000
x_train = x_train[0:total]
y_train=pickle.load(f)
f.close()
t_size = int(len(x_train)*0.7)
#x_train = x_train.reshape((119825,30,1))
#x_test = x_train[t_size:end].reshape((end-t_size, 20,1))
#y_test = y_train[t_size:end].reshape((end-t_size, 1))
#x_train = x_train[0:t_size].reshape((t_size, 20,1))
#y_train = y_train[0:t_size].reshape((t_size, 1))
x_test = x_train[t_size:]
y_test = y_train[t_size:]
print(len(x_test),len(y_test))
x_train = x_train[0:t_size]
y_train = y_train[0:t_size]
def knn(input,index):
    positive_score = 0
    negative_score = 0
    nearest = 0
    for i in range(len(x_train)):
        print(index, i)
        mie = 1000000
        d = 0
        for j in range(5):
            for k in range(30):
                mi = 100000
                for l in range(30):
                    mi = min(mi,(x_train[i][l][j]-input[k][j])**2)
                d += mi
        if(mie < d):
            mie = d
            nearest = i
    return y_test[nearest]

def weighted_majority(gamma, input, index):
    positive_score = 0
    negative_score = 0
    for i in range(len(x_train)):
        print(index,i)
        d = 0
        for j in range(5):
            for k in range(30):
                mi = 100000
                for l in range(30):
                    mi = min(mi,(x_train[i][l][j]-input[k][j])**2)
                d += mi
        if(y_train[i]==1):positive_score+=math.exp(-gamma*d)
        else: negative_score+=math.exp(-gamma*d)
    return(positive_score>negative_score)

acc = 0
for i in range(len(x_test)):
    result = weighted_majority(200,x_test[i], i)
    print(result)
    if(result == y_test[i]): acc += 1
    print(i)

print(acc/len(x_test))

#mm_scaler = preprocessing.MinMaxScaler()
#x_train = mm_scaler.fit_transform(x_train)
#print(len(x_train),len(y_train))

