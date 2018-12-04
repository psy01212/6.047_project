# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 18:10:15 2018

@author: luisxavierramostormo
"""
import numpy as np

import keras
from keras import Sequential
from keras import backend as K
from keras.layers import Dense, Conv2D, MaxPooling2D, SpatialDropout2D, Flatten
from keras.layers import BatchNormalization, Dropout

# keras docs: https://keras.io

from wig_extractor import load_training_data


class chrom_CNN(object):
    def __init__(self):
        self.batch_size = 32
        self.epochs = 12
        
        # input dimensions
        self.img_rows, self.img_cols = 1, 1000 # second number can be any
        
        
        self.all_data = load_training_data("H3K27me3", "H3K36me3", split = self.img_cols)
        first = int(.6*len(self.training_data))
        second = int(.8*len(self.training_data))
        self.x_train, self.y_train = self.all_data()[0][:first], self.all_data()[1][:first]
        self.x_val, self.y_val = self.all_data()[0][first:second], self.all_data()[1][first:second]
        self.x_test, self.y_test = self.all_data()[0][second:], self.all_data()[1][second:]

        
        
        if K.image_data_format() == 'channels_first':
            self.x_train = self.x_train.reshape(self.x_train.shape[0], 1, self.img_rows, self.img_cols)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], 1, self.img_rows, self.img_cols)
            self.input_shape = (1, self.img_rows, self.img_cols)
        else:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], self.img_rows, self.img_cols, 1)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], self.img_rows, self.img_cols, 1)
            self.input_shape = (self.img_rows, self.img_cols, 1)

        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        print('x_train shape:', self.x_train.shape)
        print(self.x_train.shape[0], 'train samples')
        print(self.x_test.shape[0], 'test samples')
        
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(1, 11),
                         activation='relu',
                         input_shape=self.input_shape))
        self.model.add(Conv2D(64, (1, 11), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(1, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(200, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.img_cols, activation='tanh')) # tanh is the kind of output we want. Between 0 and 1 and stronger gradients than sigmoid
        
        def mean_dist(y_true, y_pred):
            return K.mean(abs(y_true - y_pred))
        
        
        self.model.compile(loss='mean_squared_error',
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy', mean_dist])
        
    def train(self):
        self.model.fit(self.x_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=1,
                  validation_data=(self.x_val, self.y_val))
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        print('Test mean_dist:', score[2])

if __name__ == "__main__":
    cnn = chrom_CNN()
    cnn.train()
    
