from keras import datasets
import numpy as np
import matplotlib.pyplot as plt
import keras

# data pre-processing function
def cifar10_data_load(data, norm_data=1):
    (X_train, Y_train), (X_test, Y_test) = data.load_data()


    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')


    Y_train = keras.utils.to_categorical(Y_train)
    Y_test = keras.utils.to_categorical(Y_test)

    print('Data type changed to float32, label type changed to categorical')
    # normalize data
    if norm_data == 1:
        X_train, X_test = normalize(X_train, X_test)
        print('Normalize Data')
    else:
        pass
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

# plot history for accuracy
def plot_acc(history, title = None):
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    if title is not None:
        plt.title(title)

    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    plt.legend(['Train', 'Val'], loc=0)

# plot history for loss
def plot_loss(history, title = None):
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if title is not None:
        plt.title(title)

    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.legend(['Train', 'Val'], loc=0)

# show cifar images 4XN, if normalize is False
# dataX : image data, image labels, number of images
# display original data, not pre-processing data
def show_images(dataX, dataY, num_images):

    labels =  ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    fig = plt.figure(figsize = (10,5))

    for i in range(num_images):
        snap = fig.add_subplot(int(round(num_images/4)),4, i + 1)
        snap.set_xticks([])
        snap.set_yticks([])


        snap.set_title('{}'.format(labels[int(dataY[i])]))
        # change 'float' to 'int', reshape to show image
        plt.imshow(dataX[i].reshape(32, 32, 3))
