from __future__ import print_function
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, Flatten, Reshape, Permute
from keras.layers import Embedding, TimeDistributed
from keras.layers import LSTM, GRU, SimpleRNN, Dense, Activation, merge
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Conv2D, MaxPooling2D
from keras.applications import InceptionV3, VGG19
#from keras.layers.extra import TimeDistributedConvolution2D, TimeDistributedMaxPooling2D, TimeDistributedFlatten
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.datasets import imdb
from keras.models import Model
import tensorflow as tf
import numpy as np
import pickle
from sklearn import preprocessing
f = open("data1.pkl","rb")
x_train= pickle.load(f)
f.close()
f = open("label1.pkl","rb")

x_train = x_train[0:488476]
x_train = x_train.reshape((488476,30,1,5))


y_train=pickle.load(f)
f.close()
#mm_scaler = preprocessing.MinMaxScaler()
#x_train = mm_scaler.fit_transform(x_train)
#print(len(x_train),len(y_train))
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
#y_train = keras.utils.to_categorical(y_train, 2)
#y_test = keras.utils.to_categorical(y_test, 2)
# Embedding

model = Sequential()
'''
model.add(Conv2D(16
                 , kernel_size=(3,1),activation='relu', input_shape=(30,1,5)))
#model.add(Reshape((16,4,29)))
model.add(MaxPooling2D(pool_size=(2,1),name='pooling'))
#inter_layer = Model(inputs=model.input, outputs=model.get_layer('pooling').output)
#print(inter_layer.predict(np.zeros((3,30,1,5))).shape)
#model.add(Reshape((15,3)))

model.add(Permute((3,1,2)))
model.add(Reshape((16,14)))
model.add(Dropout(0.2))
'''
def atan(x):
    return tf.atan(x)


model = Sequential()

model.add(Conv2D(32, (3, 1), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Dropout(0.25))
'''
model.add(Conv2D(64, (3, 1), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Dropout(0.25))
'''
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=2048,
          epochs=10,
          verbose=1,
          validation_data=(x_test, y_test))

'''
model.add(LSTM(256, activation=atan, dropout_W= 0.2, dropout_U= 0.1,input_shape=(60,5)))
model.add(Dense(64,activation='sigmoid'))
model.add(Dense(16,activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=20,
          validation_data=(x_test, y_test))
'''