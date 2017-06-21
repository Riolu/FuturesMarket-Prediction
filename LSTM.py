from __future__ import print_function
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input
from keras.layers import Embedding
from keras.layers import LSTM, GRU, SimpleRNN, Dense, Activation, merge
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import imdb
from keras.models import Model
import tensorflow as tf
import numpy as np
import pickle
from sklearn import preprocessing
f = open("data1.pkl","rb")
x_train= pickle.load(f)
f.close()
x_train = x_train[0:488476]

f = open("label1.pkl","rb")
y_train=pickle.load(f)
f.close()

f = open("data0v2.pkl","rb")
x_test= pickle.load(f)[0:119815]
f.close()

f = open("label0v2.pkl","rb")
y_test=pickle.load(f)
f.close()
#mm_scaler = preprocessing.MinMaxScaler()
#x_train = mm_scaler.fit_transform(x_train)
#print(len(x_train),len(y_train))
'''
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
'''

# Embedding
max_features = 20000
maxlen = 30
embedding_size = 128

# Convolution
kernel_size = 3
filters = 16
pool_size = 4
acc = 0

def atan(x):
    return tf.atan(x)

lstm_input = Input(shape=(30,5),name='lstm_input')
lstm_output = LSTM(256, activation=atan, dropout_W= 0.2, dropout_U= 0.1)(lstm_input)
#Dense_output_0 = Dense(128,activation='sigmoid')(lstm_output)
Dense_output_1 = Dense(64,activation='sigmoid')(lstm_output)
Dense_output_2 = Dense(16,activation='sigmoid')(Dense_output_1)
predictions = Dense(1, activation='sigmoid')(lstm_output)
model = Model(input=lstm_input,outputs=predictions)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=1024,
          epochs=20,
          validation_data=(x_test, y_test))