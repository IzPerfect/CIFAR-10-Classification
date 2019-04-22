from keras import datasets
import numpy as np
import matplotlib.pyplot as plt
import keras

#  data pre-processing function
def cifar10_data_load(data):
    (X_train, Y_train), (X_test, Y_test) = data.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    Y_train = keras.utils.to_categorical(Y_train)
    Y_test = keras.utils.to_categorical(Y_test)

    # normalize data
    X_train, X_test = normalize(X_train, X_test)
    #X_train /= 255.
    #X_test /= 255.

    L, W, H, C = X_train.shape
    X_train = X_train.reshape(-1, W*H*C)
    X_test = X_test.reshape(-1, W*H*C)


    return (X_train, Y_train), (X_test, Y_test)

# normalization of data
def normalize(X_train, X_test):
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))

    X_train = (X_train - mean)/(std + 1e-7)
    X_test = (X_test - mean)/(std + 1e-7)

    return X_train, X_test
