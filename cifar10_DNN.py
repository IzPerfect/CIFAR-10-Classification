import keras
from keras import models
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from data_utils import *
import os





class CifarDNN(object):
    def __init__(self, img_shape, class_num, actf = 'relu', learning_rate = 0.001, layer1 = 32, layer2 = 64,  drop_rate = 0.2):

        self.learning_rate = learning_rate
        self.actf = actf
        self.layer1 = layer1
        self.layer2 = layer2
        self.img_shape = img_shape
        self.class_num = class_num
        self.drop_rate = drop_rate


        self.model = self.build_model()

    # build network to classify cifar data
    def build_model(self):
        model = models.Sequential()
        model.add(Dense(self.layer1, input_shape = (self.img_shape,), name = "HiddenLayer1"))
        model.add(Activation(self.actf))
        model.add(Dropout(self.drop_rate))

        model.add(Dense(self.layer2, name = "HiddenLayer2"))
        model.add(Activation(self.actf))
        model.add(Dropout(self.drop_rate))

        model.add(Dense(self.class_num))
        model.add(Activation('softmax'))

        model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = self.learning_rate)
                          ,metrics = ['accuracy'])

        return model

    # train with data augmentation
    def train_generator(self, X_train, Y_train, epoch = 10, batch_size = 32, val_split = 0.2, save_model = True):
        self.generator = ImageDataGenerator(
            featurewise_center = True,
            featurewise_std_normalization = True,
            rotation_range = 20,
            width_shift_range = 0.2,
            height_shift_range = 0.2
        )

        X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
        self.generator.fit(X_train)

        save_dir = './save_model/'
        if not os.path.exists(save_dir): # if there is no exist, make the path
            os.makedirs(save_dir)

        model_path = save_dir + '{epoch:02d}-{val_loss:.4f}-{acc:.4f}.hd5'
        cb_checkpoint = ModelCheckpoint(filepath = model_path, monitor = 'val_loss',
            verbose = 1, save_best_only = True)


        #X_train = X_train.reshape(-1, 32*32*3)

        self.history_generator = self.model.fit_generator(self.generator.flow(X_train, Y_train,
                                         batch_size=batch_size),
                            steps_per_epoch = X_train.shape[0] // batch_size,
                            epochs = epoch,
                            callbacks=[cb_checkpoint])

        return self.history_generator


    # train model
    def train(self, X_train, Y_train, epoch = 10, batch_size = 32, val_split = 0.2, save_model = True):

        save_dir = './save_model/'
        if not os.path.exists(save_dir): # if there is no exist, make the path
            os.makedirs(save_dir)

        model_path = save_dir + '{epoch:02d}-{val_loss:.4f}-{acc:.4f}.hd5'
        cb_checkpoint = ModelCheckpoint(filepath = model_path, monitor = 'val_loss',
            verbose = 1, save_best_only = True)

        self.history = self.model.fit(X_train, Y_train, epochs = epoch,
                                          batch_size = batch_size, validation_split = val_split,
                                            callbacks = [cb_checkpoint])


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
