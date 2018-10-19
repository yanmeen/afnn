# -*- coding: utf-8 -*-

# by Dr. Ming Yan (10/2018)
# yan.meen@gmail.com
# https://github.com/afnn
# modified on the code from https://github.com/cszn
# =============================================================================

# run this to train the model

# =============================================================================
# For batch normalization layer, momentum should be a value from [0, 0.9] rather than the default
# =============================================================================

import argparse
import re
import os
import glob
import datetime
import numpy as np
from keras.layers import Input, Conv2D, BatchNormalization, Activation
from keras.layers import Dense, MaxPooling2D, Concatenate, Flatten, Dropout
from keras import losses
from keras.models import Model, load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import optimizers
from keras.utils import np_utils
import data_generator as dg
#import keras.backend as K

# Params
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='AFNN', type=str,
                    help='choose a type of model')
parser.add_argument('--batch_size', default=160, type=int, help='batch size')
parser.add_argument('--train_data', default='data/SMRes128',
                    type=str, help='path of train data')
parser.add_argument('--epoch', default=2000, type=int,
                    help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='initial learning rate for Adam')
parser.add_argument('--save_step', default=10, type=int,
                    help='save model at every x epoches')
args = parser.parse_args()


save_dir = os.path.join('models', args.model+'_'+'128')

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# Network for focal plane calculation using specialized light path
# training with only few image stacks
# source image stack is 128x128, 200 focal planes with 1um step
# Network use conv layers with Relu and MaxPooling constract features
# use dense layers, dropout layer with softmax as output layes
# first Conv2D layer:
#	64 filters
#	[3x3], stride 2x2
#	ReLU output [64x64] =================   to path2
#	MaxPooling(2x2) output [32x32]
#	[3x3], stride 2x2
#	ReLU output [16x16]
#	MaxPooling(2x2) [8x8]
#	[3x3], stride 2x2
#	ReLU output [4x4]
#	MaxPooling(2x2)
#	output [2x2] ================   to merge layer
# second Conv2D layer:
#	[7x7], stride 4x4
#	ReLU output [32x32] ===================   to path3
#	MaxPooling(2x2) [16x16]
#	[5x5], stride 2x2
#	ReLU [8x8]
#	MaxPooling(2x2)
#	output [4x4] ================   to merge layer
# third Conv2D layer:
#	[7x7], stride 4x4
#	ReLU output [8x8]
#	MaxPooling(2x2)
#	output [4x4] ================   to merge layer
# Merge the Conv2D layer outputs for dense, first concatenate then flatten
#	64x[4x4] *3
# =========================================================================
# solution 2:
# Dense Layer constract the features into 64
#	ReLU
#	dropout(0.5)
# Dense Layer constract into 1
# use L2 loss


def AFNN(filters=8, image_channels=1, use_bnorm=True):
    layer_count = 0
    inpt = Input(shape=(128, 128, image_channels),
                 name='input'+str(layer_count))
    # 1st layer, Conv+relu
    layer_count += 1
    x_0 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1),
                 kernel_initializer='Orthogonal', padding='same',
                 name='conv'+str(layer_count))(inpt)
    layer_count += 1
    x_0 = Activation('relu', name='relu'+str(layer_count))(x_0)
    # Path 1
    layer_count += 1
    x1 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None,
                      name='maxpool_p1'+str(layer_count))(x_0)
    # 2 layers, Conv+BN+relu+MaxPooling
    for i in range(2):
        layer_count += 1
        x1 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(2, 2),
                    kernel_initializer='Orthogonal', padding='same', use_bias=False,
                    name='conv_p1'+str(layer_count))(x1)
        if use_bnorm:
            layer_count += 1
            x1 = BatchNormalization(
                axis=3, momentum=0.0, epsilon=0.0001, name='bn_p1'+str(layer_count))(x1)
        layer_count += 1
        x1 = Activation('relu', name='relu_p1'+str(layer_count))(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same',
                          data_format=None, name='maxpool_p1'+str(layer_count))(x1)

    # Path 2
    # 2 layers, Conv+BN+ReLU+MaxPooling
    layer_count += 1
    x_1 = Conv2D(filters=filters, kernel_size=(7, 7), strides=(4, 4),
                 kernel_initializer='Orthogonal', padding='same',
                 name='conv_p2'+str(layer_count))(x_0)
    layer_count += 1
    x_1 = Activation('relu', name='relu_p2'+str(layer_count))(x_1)

    # Path 2_1
    # 1 layer, Conv+BN+ReLU+MaxPooling
    layer_count += 1
    x2 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same',
                      data_format=None, name='maxpool_p2'+str(layer_count))(x_1)
    layer_count += 1
    x2 = Conv2D(filters=filters, kernel_size=(5, 5), strides=(2, 2),
                kernel_initializer='Orthogonal', padding='same', use_bias=False,
                name='conv_p2'+str(layer_count))(x2)
    if use_bnorm:
        layer_count += 1
        x2 = BatchNormalization(
            axis=3, momentum=0.0, epsilon=0.0001, name='bn_p2'+str(layer_count))(x2)
    layer_count += 1
    x2 = Activation('relu', name='relu_p2'+str(layer_count))(x2)
    x2 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None,
                      name='maxpool_p2'+str(layer_count))(x2)

    # Path 2_2
    # 1 layer, Conv+BN+ReLU+MaxPooling
    layer_count += 1
    x_2 = Conv2D(filters=filters, kernel_size=(7, 7), strides=(4, 4),
                 kernel_initializer='Orthogonal', padding='same',
                 name='conv_p3'+str(layer_count))(x_1)
    if use_bnorm:
        layer_count += 1
        x_2 = BatchNormalization(
            axis=3, momentum=0.0, epsilon=0.0001, name='bn_p3'+str(layer_count))(x_2)
    layer_count += 1
    x3 = Activation('relu', name='relu_p3'+str(layer_count))(x_2)
    layer_count += 1
    x3 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None,
                      name='maxpool_p3'+str(layer_count))(x3)

    # Merge layer
    layer_count += 1
    x = Concatenate(axis=-1, name='concat'+str(layer_count))([x1, x2, x3])
    layer_count += 1
    x = Flatten(data_format=None, name='Flat'+str(layer_count))(x)

    # Dense output layer
    layer_count += 1
    x = Dense(100, activation='relu', name='dense'+str(layer_count))(x)
    layer_count += 1
    x = Dropout(0.5, name='dropout'+str(layer_count))(x)
    layer_count += 1
    y = Dense(50, activation='softmax', name='dense'+str(layer_count))(x)
    model = Model(inputs=inpt, outputs=y)

    return model


def find_LastCheckpoint(model_dir):
    # get name list of all .hdf5 files
    file_list = glob.glob(os.path.join(model_dir, 'model_*.hdf5'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).hdf5.*", file_)
            # print(result[0])
            epochs_exist.append(int(result[0]))
        init_epoch = max(epochs_exist)
    else:
        init_epoch = 0
    return init_epoch


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def lr_schedule(epoch):
    initial_lr = args.lr
    if epoch <= 40:
        lr = initial_lr
    elif epoch <= 100:
        lr = initial_lr/10
    elif epoch <= 200:
        lr = initial_lr/20
    else:
        lr = initial_lr/20
    log('current learning rate is %2.8f' % lr)
    return lr


if __name__ == '__main__':
    # model selection
    AF_model = AFNN(filters=4, image_channels=1, use_bnorm=True)
    AF_model.summary()

    # load the last model in matconvnet style
    initial_epoch = find_LastCheckpoint(model_dir=save_dir)
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        AF_model = load_model(os.path.join(
            save_dir, 'model_%03d.hdf5' % initial_epoch), compile=False)

    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    ad = optimizers.Adam(0.01)
    # compile the model
    AF_model.compile(optimizer=ad, metrics=['accuracy'],
                     loss=losses.categorical_crossentropy)

    # use call back functions
    check_pointer = ModelCheckpoint(os.path.join(save_dir, 'model_{epoch:03d}.hdf5'),
                                    verbose=1, save_weights_only=False, period=args.save_step)
    csv_logger = CSVLogger(os.path.join(
        save_dir, 'log.csv'), append=True, separator=',')
    lr_scheduler = LearningRateScheduler(lr_schedule)
    tensor_board = TensorBoard(
        "./logs", histogram_freq=5, batch_size=160, write_graph=True, write_images=False
    )
    xs, ys = dg.datagenerator(data_dir=args.train_data)
    xs = xs.astype('float32')
    xs = xs/255
    ys = np_utils.to_categorical(ys)

    history = AF_model.fit(xs, ys, batch_size=args.batch_size, epochs=args.epoch, verbose=1, validation_split=0.05,
                           initial_epoch=initial_epoch, shuffle=True,
                           callbacks=[check_pointer, csv_logger, tensor_board])
