import keras
from keras import models
from keras.layers import Dense, Dropout, Flatten, Activation, Lambda, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from data_utils import *
import os
import time

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

class CifarVGG(object):
    def __init__(self, img_shape, class_num, actf = 'relu',
        learning_rate = 0.001,  drop_rate = 0.2, do_batch_norm = False, do_drop = False):

        self.learning_rate = learning_rate
        self.actf = actf
        self.img_shape = img_shape
        self.class_num = class_num
        self.drop_rate = drop_rate
        self.do_batch_norm = do_batch_norm
        self.do_drop = do_drop

        self.build_model()

    def cifar10_normalize(self, x):
        # cifar10 training set normalization
        mean = 120.707
        std = 64.15
        #x = x / 255.
        return (x-mean)/(std+1e-7)

    def cifar_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2])

    def fc_block(self, layer_num):
        self.model.add(Dense(layer_num, activation = self.actf))
        self.model.add(Dropout(self.drop_rate)) if self.do_drop else None



    def conv_block(self, layers, feature_maps, filter_size = (3, 3),
        conv_strides = 1, pooling_filter_size = (2, 2), pooling_strides = (2, 2)):

        for i in range(layers):
            self.model.add(Conv2D(feature_maps, filter_size, padding = 'same', activation = self.actf))
            self.model.add(BatchNormalization()) if self.do_batch_norm else None

        self.model.add(MaxPooling2D(pooling_filter_size, strides = pooling_strides))

    # build network to classify cifar data
    def build_model(self):
        self.model = models.Sequential()

        self.model.add(Lambda(self.cifar10_normalize, input_shape = self.img_shape))
        #self.model.add(Lambda(lambda x: x / 255.0, input_shape = self.img_shape))
        #self.model.add(Conv2D(64, (3, 3), padding='same', input_shape= self.img_shape))

        self.conv_block(layers = 2, feature_maps = 64)
        self.conv_block(layers = 2, feature_maps = 128)
        self.conv_block(layers = 3, feature_maps = 256)
        self.conv_block(layers = 3, feature_maps = 512)
        self.conv_block(layers = 3, feature_maps = 512)

        self.model.add(Flatten())

        self.fc_block(4096)
        self.fc_block(4096)


        self.model.add(Dense(self.class_num))
        self.model.add(Activation('softmax'))

        self.model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = self.learning_rate)
                          ,metrics = ['accuracy'])



    # train model
    def train(self, X_train, Y_train, epoch = 10, batch_size = 32, val_split = 0.2, save_model = True):

        # current time
        now = time.localtime()
        time_now = "%04d-%02d-%02d_%02dh%02dm%02ds" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

        save_dir = './save_model/model/vgg_' + time_now + '/'
        if not os.path.exists(save_dir): # if there is no exist, make the path
            os.makedirs(save_dir)

        model_path = save_dir + '{epoch:02d}-{val_loss:.4f}-{acc:.4f}.hd5'
        cb_checkpoint = ModelCheckpoint(filepath = model_path, monitor = 'val_loss',
            verbose = 1, save_best_only = True)

        self.history = self.model.fit(X_train, Y_train, epochs = epoch,
                                          batch_size = batch_size, validation_split = val_split,
                                            callbacks = [cb_checkpoint])


        save_dir_history = './save_model/history/vgg_'+ time_now + '/'
        if not os.path.exists(save_dir_history): # if there is no exist, make the path
            os.makedirs(save_dir_history)


        fig_acc = plt.figure(1)
        plt.plot(self.history.history['acc'])
        fig_acc.savefig(save_dir_history + 'acc_history' + '.jpg')
        np.save(save_dir_history + 'acc_history' + '.npy', self.history.history['acc'])

        fig_loss = plt.figure(2)
        plt.plot(self.history.history['loss'])
        fig_acc.savefig(save_dir_history + 'loss_history' + '.jpg')
        np.save(save_dir_history + 'loss_history' + '.npy', self.history.history['loss'])


        return self.history

    def saved_model_use(self, save_dir = None):
        if save_dir == None:
            return print('No path')

        self.model.load_weights(save_dir)

        return print("Loaded model from '{}'".format(save_dir))

    # evalutate model
    def show_eval(self, X_test, Y_test, batch_test_size = 10):
        self.score = self.model.evaluate(X_test, Y_test, batch_size = batch_test_size)
        print('Test Loss : ', self.score[0])
        print('Test Accuracy : ', self.score[1])

        return self.score

    def predict(self, X_test):
        pred_classes = self.model.predict(X_test)
        pred_classes = np.argmax(pred_classes, axis = 1)

        return pred_classes

    def show_model(self):
        return print(self.model.summary())
