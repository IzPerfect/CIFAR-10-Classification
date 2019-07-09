import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dropout\
    ,GlobalAveragePooling2D, Dense, BatchNormalization, Activation, DepthwiseConv2D, add, Reshape
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

import matplotlib.pyplot as plt
import numpy as np
from data_utils import *
import os
import time


class MobileNetV2(object):
    def __init__(self, img_shape, class_num,
        learning_rate = 0.001,  drop_rate = 0.5, do_drop = True, weight_decay = 0.01):

        self.learning_rate = learning_rate
        self.img_shape = img_shape
        self.class_num = class_num
        self.drop_rate = drop_rate
        self.do_drop = do_drop
        self.weight_decay = weight_decay


        self.model = self.build_model(self.class_num)


    def conv_block(self, x, c_out, kernel_size, s, actf, name, weight_decay = 0):
        '''
        convolution block
            Can be 'Expansion' or 'projection' convolution layer block
            Convolution - BatchNormalization - ReLU6(or not)
        '''
        x = Conv2D(c_out , kernel_size, strides = s,
                        padding = 'same', kernel_regularizer=regularizers.l2(weight_decay), name = name + '_conv')(x)
        x = BatchNormalization(name = name + '_conv_batchnorm')(x)

        if actf:
            x = Activation(lambda x: K.relu(x, max_value = 6.0), name = name + '_conv_relu6')(x)

        return x

    def depthwise_conv_block(self, x, depth_conv_strides, name, weight_decay = 0, kernel_size = (3, 3)):
        '''
        depthwise convolution block
            depthwise convolution block
            Depthwise Convoltuion(3x3) - BatchNormalization - ReLU6
        '''
        x = DepthwiseConv2D(kernel_size, strides = depth_conv_strides, depth_multiplier = 1,
            padding = 'same', kernel_regularizer=regularizers.l2(weight_decay), name = name + '_depthwise_conv')(x)
        x = BatchNormalization(name = name + '_depth_conv_batchnorm')(x)
        x = Activation(lambda x: K.relu(x, max_value = 6.0), name = name + '_depth_conv_relu6')(x)

        return x


    def inverted_residual_block(self, x, expanded_channels, c_out, depth_kernel_size, depth_conv_strides, weight_decay, name, kernel_size = (1, 1)
        , conv_strides = (1, 1)):
        '''
        inverted residual block
            Expansion layer - depthwise layer - projection layer
            narrow - wide - narrow approach
            Inverted Residuals were used. If channels are same and stride == (1, 1), add.
        '''
        in_channels = K.int_shape(x)[-1]

        expansion_layer = self.conv_block(x, expanded_channels, kernel_size, conv_strides, actf = True, weight_decay = weight_decay, name = name + '_expansion')
        depthwise_layer = self.depthwise_conv_block(expansion_layer, depth_conv_strides = depth_conv_strides, weight_decay = weight_decay, name = name + '_depthwise')
        projection_layer = self.conv_block(depthwise_layer, c_out, kernel_size, conv_strides, actf = False, weight_decay = weight_decay, name = name + '_projection')

        # Check the number of channels and strides to add
        if in_channels == c_out and depth_conv_strides == (1, 1):
            x = add([x, projection_layer])
        else:
            x = projection_layer
        return x


    def bottleneck(self, x, depth_kernel_size, t, c, n, s, weight_decay, name):
        '''
            This function defines a sequence of 1 or more inverted residual block blocks
        '''
        in_channels = K.int_shape(x)[-1]
        x = self.inverted_residual_block(x, t*in_channels, c, depth_kernel_size = depth_kernel_size, depth_conv_strides = s,
            weight_decay = weight_decay, name = name + str(0))

        for i in range(1, n):
            in_channels = K.int_shape(x)[-1]
            x = self.inverted_residual_block(x, t*in_channels, c, depth_kernel_size = depth_kernel_size, depth_conv_strides = (1, 1),
                weight_decay = weight_decay, name = name + str(i))
        return x


    def build_model(self, k):
        inputs = Input(self.img_shape)

        x = self.conv_block(inputs, c_out = 32, kernel_size = (3, 3), s = (2, 2), actf = True, name = 'conv2d_layer1')

        x = self.bottleneck(x, depth_kernel_size = (3, 3), t = 1, c = 16, n = 1, s = (1, 1), weight_decay = self.weight_decay, name = 'bottleneck_1')
        x = self.bottleneck(x, depth_kernel_size = (3, 3), t = 6, c = 24, n = 2, s = (2, 2), weight_decay = self.weight_decay, name = 'bottleneck_2')
        x = self.bottleneck(x, depth_kernel_size = (3, 3), t = 6, c = 32, n = 3, s = (2, 2), weight_decay = self.weight_decay, name = 'bottleneck_3')
        x = self.bottleneck(x, depth_kernel_size = (3, 3), t = 6, c = 64, n = 4, s = (2, 2), weight_decay = self.weight_decay, name = 'bottleneck_4')
        x = self.bottleneck(x, depth_kernel_size = (3, 3), t = 6, c = 96, n = 3, s = (1, 1), weight_decay = self.weight_decay, name = 'bottleneck_5')
        x = self.bottleneck(x, depth_kernel_size = (3, 3), t = 6, c = 160, n = 3, s = (2, 2), weight_decay = self.weight_decay, name = 'bottleneck_6')
        x = self.bottleneck(x, depth_kernel_size = (3, 3), t = 6, c = 320, n = 1, s = (1, 1), weight_decay = self.weight_decay, name = 'bottleneck_7')

        x = self.conv_block(x, c_out = 1280, kernel_size = (1, 1), s = (1, 1), actf = False, name = 'conv2d_layer2')

        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, -1))(x)

        if self.do_drop:
            x = Dropout(self.drop_rate)(x)

        x = self.conv_block(x, c_out = k, kernel_size = (1, 1), s = (1, 1), actf = False, name = 'conv2d_last')
        x = Activation('softmax')(x)
        x = Reshape((k,))(x)

        model = Model(inputs = inputs, outputs = x)
        model.compile(loss='categorical_crossentropy', optimizer= Adam(lr = self.learning_rate), metrics=['accuracy'])
        return model

    # train model
    def train(self, X_train, Y_train, epoch = 10, batch_size = 32, val_split = 0.2, save_model = True, aug_data = False):

        # current time
        now = time.localtime()
        time_now = "%04d-%02d-%02d_%02dh%02dm%02ds" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

        save_dir = './save_model/model/mobilenetv2_' + time_now + '/'
        if not os.path.exists(save_dir): # if there is no exist, make the path
            os.makedirs(save_dir)

        model_path = save_dir + '{epoch:02d}-{val_loss:.4f}-{acc:.4f}.hd5'
        cb_checkpoint = ModelCheckpoint(filepath = model_path, monitor = 'val_loss',
            verbose = 1, save_best_only = True)

        reduce_lr = ReduceLROnPlateau(monitor = 'val_acc', factor = 0.2, patience = 10, verbose = 1, min_lr = 1e-06)

        start_time = time.time()

        if (aug_data):
            print('Data augmentation and Train')
            data_generator = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True)

            val_data_generator = ImageDataGenerator()

            # To split train, validation data
            split_range = np.ceil(X_train.shape[0]-X_train.shape[0]*val_split).astype(np.int)
            train_dataX = X_train[:]
            train_dataY = Y_train[:]

            X_train, X_val = train_dataX[:split_range], train_dataX[split_range:]
            Y_train, Y_val = train_dataY[:split_range], train_dataY[split_range:]


            data_generator.fit(X_train)
            val_data_generator.fit(X_val)


            train_generator = data_generator.flow(
                X_train,
                Y_train,
                batch_size = batch_size)

            val_generator = val_data_generator.flow(
                X_val,
                Y_val,
                batch_size = batch_size)

            self.history = self.model.fit_generator(train_generator,
                steps_per_epoch = X_train.shape[0] // batch_size,
                epochs = epoch,
                validation_data = val_generator,
                validation_steps = X_val.shape[0] // batch_size,
                callbacks = [cb_checkpoint, reduce_lr])

        else:
            self.history = self.model.fit(X_train, Y_train, epochs = epoch,
                                              batch_size = batch_size, validation_split = val_split,
                                                callbacks = [cb_checkpoint, reduce_lr])

        print("\n Training --- %s sec---" %(time.time() - start_time))


        now = time.localtime()
        time_now = "%04d-%02d-%02d_%02dh%02dm%02ds" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

        save_dir_history = './save_model/history/mobilenetv2_'+ time_now + '/'
        if not os.path.exists(save_dir_history): # if there is no exist, make the path
            os.makedirs(save_dir_history)


        fig_acc = figure_acc(self.history)
        fig_acc.savefig(save_dir_history + 'acc_history' + '.jpg')
        np.save(save_dir_history + 'acc_history' + '.npy', self.history.history['acc'])

        fig_loss = figure_loss(self.history)
        fig_loss.savefig(save_dir_history + 'loss_history' + '.jpg')
        np.save(save_dir_history + 'loss_history' + '.npy', self.history.history['loss'])

        plt.close('all')


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
