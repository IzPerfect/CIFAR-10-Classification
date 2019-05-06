import keras
from keras import datasets
import argparse
import os
from data_utils import *
from cifar10_DNN import *
from cifar10_VGG import *

# get bool type
# Because when enter command, that command is str
def str2bool(args):
    if args in ('True'):
        return True
    else:
        return False


# get arguments
def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', dest = 'epoch', type = int,default = 10, help ='training epoch (default : 10)')
    parser.add_argument('--batch_size', dest = 'batch_size', type = int, default = 16, help ='training batch_size (default : 16)')
    parser.add_argument('--learning_rate', dest = 'learning_rate', type = float, default = 0.001, help ='training learning rate (default : 0.001)')
    parser.add_argument('--actf', dest = 'actf', default = 'relu', help ='training activation function (default : relu)')
    parser.add_argument('--layer1', dest = 'layer1', type = int, default = 32, help ='hidden layer1 (default : 32)')
    parser.add_argument('--layer2', dest = 'layer2', type = int, default = 64, help ='hidden layer2 (default : 64)')
    parser.add_argument('--drop_rate', dest = 'drop_rate', type = float, default = 0.2, help ='drop rate (default : 0.2)')
    parser.add_argument('--val_split', dest = 'val_split', type = float, default = 0.2, help ='validation split rate (default : 0.2)')
    parser.add_argument('--std_data', dest = 'std_data', type = str2bool, default = True, help ='standardization data ,another number:use raw data (default : True)')
    parser.add_argument('--do_drop', dest = 'do_drop', type = str2bool, default = True, help ='Drop out(default : True)')
    parser.add_argument('--do_batch_norm', dest = 'do_batch_norm', type = str2bool, default = True, help ='BatchNormalization(default : True)')
    parser.add_argument('--aug_data', dest = 'aug_data', type = str2bool, default = True, help ='Augmentation Data(default : True)')

    parser.add_argument('--batch_test_size', dest = 'batch_test_size', type = int, default = 32, help ='batch_test_size (default : 32)')

    return parser.parse_args()

# main function
def main(args):
    (X_train, Y_train), (X_test, Y_test) = cifar_DNN_data_load(datasets.cifar10, args.std_data)
    cifar_model = CifarDNN(img_shape = X_train.shape[1], class_num = Y_train.shape[1],  learning_rate = args.learning_rate,
            actf = args.actf, layer1 = args.layer1, layer2 = args.layer2, drop_rate = args.drop_rate, do_drop = args.do_drop)

    # model train
    history_train = cifar_model.train(X_train, Y_train, epoch = args.epoch,
        batch_size = args.batch_size, val_split = args.val_split)

    # model evaluate
    result = cifar_model.show_eval(X_test, Y_test, args.batch_test_size)
    print('Test Loss : ', result[0])
    print('Test Accuracy : ', result[1]*100, '%')


    # model history plot - loss and accuracy
    plt.close('all') # close all figure when training

    plot_loss(history_train)
    plot_acc(history_train)
    plt.show()

def main_vgg(args):
    (X_train, Y_train), (X_test, Y_test) = cifar_VGG_data_load(datasets.cifar10, args.std_data)
    cifar_model = CifarVGG(img_shape = X_train[0].shape, class_num = Y_train.shape[1],
                                      do_batch_norm = args.do_batch_norm, do_drop = args.do_drop)
    history_train = cifar_model.train(X_train, Y_train, epoch = args.epoch,
        batch_size = args.batch_size, val_split = args.val_split, aug_data = args.aug_data)

    result = cifar_model.show_eval(X_test, Y_test, 32)
    print('Test Loss : ', result[0])
    print('Test Accuracy : ', result[1]*100, '%')

    # model history plot - loss and accuracy
    plt.close('all') # close all figure when training

    plot_loss(history_train)
    plot_acc(history_train)
    plt.show()

if __name__ == '__main__':

    args = arg_parser()
    print("Args : ", args)

    # run main function
    #main(args)
    main_vgg(args)
