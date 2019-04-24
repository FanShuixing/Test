import os
import keras
import tensorflow as tf
print(tf.__version__)
from tensorflow.contrib.training import HParams
#from module.module import BuildModel, DataPrepare
import six
from keras.models import Sequential
from keras.models import Model
from keras.layers import *
from keras.layers.convolutional import *
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

filter = 32

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(512,512,23)))
    model.add(Convolution2D(filter, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(filter, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(filter*2, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(filter*2, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(filter*4, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(filter*4, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(filter*4, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(filter*8, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(filter*8, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(filter*8, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(filter*8, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(filter*8, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(filter*8, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(filter*64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(filter*64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

def train(output="./logs"):
    model = keras.models.load_model('./model/vgg_3_100_mse.model')
    x_train = np.load('./train_512_fenlei123_shuffle.npy')
    y_train = np.load('./label_512_fenlei123_shuffle.npy')
    print(x_train.shape, y_train.shape)
    x_train = x_train.reshape((2080,512,512,23))
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=8)
    #model = VGG_16()
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
    model_checkpoint = ModelCheckpoint('./model/resnet_auto.h5', monitor='auto',save_best_only=True,mode= 'max')
    model.fit(x_train, y_train, batch_size=4, class_weight = 'auto',epochs=100,validation_data=(x_test, y_test), shuffle=True)
    #model.fit(x_train, y_train, batch_size=8,epochs=50,validation_data=(x_test, y_test), shuffle=True)
    model.save("./model/vgg_3_200_mse.model")
    print("Optimization Finished!")

if __name__ == '__main__':
     train()
