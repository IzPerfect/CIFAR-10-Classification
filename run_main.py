import keras
from keras import datasets
import argparse
import os
from data_utils import *
from cifar10_DNN import *

# get arguments
def arg_parser():
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

    return parser.parse_args()

# main function
def main(args):
    (X_train, Y_train), (X_test, Y_test) = cifar10_data_load(datasets.cifar10, args.normalize)
    cifar_model = CifarDNN(img_shape = X_train.shape[1], class_num = Y_train.shape[1],  learning_rate = args.learning_rate,
            actf = args.actf, layer1 = args.layer1, layer2 = args.layer2, drop_rate = args.drop_rate)

    # model train
    history_train = cifar_model.train(X_train, Y_train, epoch = args.epoch,
        batch_size = args.batch_size, val_split = args.val_split)

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

if __name__ == '__main__':

    args = arg_parser()
    print("Args : ", args)

    # run main function
    main(args)
