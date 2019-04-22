import keras
import numpy as np
from keras import datasets
from keras import models
from keras.layers import Dense, Dropout, Flatten, Activation
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import argparse
import os
from utils import *



class CifarDNN(object):
    def __init__(self, img_shape, class_num, epoch = 10, batch_size = 32, learning_rate = 0.001, actf = 'relu',
                     layer1 = 32, layer2 = 64,  drop_rate = 0.2, val_split = 0.2):
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.actf = actf
        self.layer1 = layer1
        self.layer2 = layer2
        self.img_shape = img_shape
        self.class_num = class_num
        self.drop_rate = drop_rate
        self.val_split = val_split


        self.model = self.build_model()

    # cifar 분류에 사용될 neural network 설정
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

    # 모델학습
    def train(self, X_train, Y_train):
        self.history = self.model.fit(X_train, Y_train, epochs = self.epoch,
                                          batch_size = self.batch_size, validation_split = self.val_split)
        return self.history

    # 모델평가
    def show_eval(self, X_test, Y_test, batch_test_size = 10):
        self.result = self.model.evaluate(X_test, Y_test, batch_size = batch_test_size)

        return self.result





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', dest = 'epoch', type = int,default = 10, help ='training epoch (default : 10)')
    parser.add_argument('--batch_size', dest = 'batch_size', type = int, default = 32, help ='training batch_size (default : 32)')
    parser.add_argument('--learning_rate', dest = 'learning_rate', type = float, default = 0.001, help ='training learning rate (default : 0.001)')
    parser.add_argument('--actf', dest = 'actf', default = 'relu', help ='training activation function (default : relu)')
    parser.add_argument('--layer1', dest = 'layer1', type = int, default = 32, help ='hidden layer1 (default : 32)')
    parser.add_argument('--layer2', dest = 'layer2', type = int, default = 64, help ='hidden layer2 (default : 64)')
    parser.add_argument('--drop_rate', dest = 'drop_rate', type = float, default = 0.2, help ='drop rate (default : 0.2)')
    parser.add_argument('--val_split', dest = 'val_split', type = float, default = 0.2, help ='validation split rate (default : 0.2)')
    parser.add_argument('--normalize', dest = 'normalize', type = int, default = 1, help ='1:normalize data ,another number:use raw data (default : 1(True))')

    parser.add_argument('--batch_test_size', dest = 'batch_test_size', type = int, default = 32, help ='batch_test_size (default : 32)')
    args = parser.parse_args()

    print("Args : ", args)

    (X_train, Y_train), (X_test, Y_test) = cifar10_data_load(datasets.cifar10, args.normalize)
    cifar_model = CifarDNN(img_shape = X_train.shape[1], class_num = Y_train.shape[1], epoch = args.epoch,
        batch_size = args.batch_size, learning_rate = args.learning_rate,
            actf = args.actf, layer1 = args.layer1, layer2 = args.layer2, drop_rate = args.drop_rate, val_split = args.val_split)

    # model train
    history_train = cifar_model.train(X_train, Y_train)

    # model evaluate
    result = cifar_model.show_eval(X_test, Y_test, args.batch_test_size)
    print('Test Loss : ', result[0])
    print('Test Accuracy : ', result[1]*100, '%')

    print('predict clas : ',cifar_model.model.predict_classes(X_test[:3]))
    print('actual class : ', np.where(Y_test[:3])[1])

    # model history plot - loss and accuracy
    plot_loss(history_train)
    plt.show()
    plot_acc(history_train)
    plt.show()
