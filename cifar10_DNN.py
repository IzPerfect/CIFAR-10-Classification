import keras
import numpy as np
from keras import models
from keras.layers import Dense, Dropout, Flatten, Activation
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from data_utils import *



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

    # train model
    def train(self, X_train, Y_train, epoch = 10, batch_size = 32, val_split = 0.2):
        self.history = self.model.fit(X_train, Y_train, epochs = epoch,
                                          batch_size = batch_size, validation_split = val_split)
        return self.history

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
