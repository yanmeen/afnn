# -*- coding: utf-8 -*-

# by Dr. Ming Yan (10/2018)
# yan.meen@gmail.com
# https://github.com/yanmeen/afnn
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
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
import data_generator as dg
#import keras.backend as K

# Params
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='AFNN', type=str,
                    help='choose a type of model')
parser.add_argument('--batch_size', default=200, type=int, help='batch size')
parser.add_argument('--train_data', default='data/Res128',
                    type=str, help='path of train data')
parser.add_argument('--epoch', default=500, type=int,
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
#	input [128x128], 64 filters
#	[3x3], stride 1x1
#	ReLU output
#	MaxPooling(2x2) output [64x64]
#	[3x3], stride 1x1
#	ReLU output
#	MaxPooling(2x2) [32x32]
#	[3x3], stride 1x1
#	ReLU output
#	MaxPooling(2x2) [16x16]
#	[3x3], stride 1x1
#	ReLU output
#	MaxPooling(2x2) [8x8]
#	output [8x8] ================   to Dense layer
# Flatten the Conv2D layer outputs for dense, first concatenate then flatten
#	64x[8x8]
# =========================================================================
# solution 2:
# Dense Layer constract the features into 128
#	ReLU
#	dropout(0.5)
# Dense Layer constract into 1
# use L2 loss


def AFNN(filters=8, image_channels=1, use_bnorm=True):
    layer_count = 0
    inpt = Input(shape=(128, 128, image_channels),
                 name='input_'+str(layer_count))
    # 1st layer, Conv+relu
    layer_count += 1
    x_0 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(2, 2),
                 kernel_initializer='Orthogonal', padding='same',
                 name='conv_'+str(layer_count))(inpt)
    layer_count += 1
    x_0 = Activation('relu', name='relu_'+str(layer_count))(x_0)
    # Path 1
    layer_count += 1
    x_0 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None,
                       name='maxpool_'+str(layer_count))(x_0)
    # 2 layers, Conv+BN+relu+MaxPooling
    for i in range(2):
        layer_count += 1
        x_0 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(2, 2),
                     kernel_initializer='Orthogonal', padding='same', use_bias=False,
                     name='conv_'+str(layer_count))(x_0)
        if use_bnorm:
            layer_count += 1
            x_0 = BatchNormalization(
                axis=3, momentum=0.0, epsilon=0.0001, name='bn_'+str(layer_count))(x_0)
        layer_count += 1
        x_0 = Activation('relu', name='relu_'+str(layer_count))(x_0)
        x_0 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same',
                           data_format=None, name='maxpool_'+str(layer_count))(x_0)

    # Dense  layers
    layer_count += 1
    x_0 = Flatten(data_format=None, name='Flat_'+str(layer_count))(x_0)

    # Dense output layer
    layer_count += 1
    x_0 = Dense(200, activation='relu', name='dense_'+str(layer_count))(x_0)
    layer_count += 1
    x_0 = Dropout(0.5, name='dropout_'+str(layer_count))(x_0)
    layer_count += 1
    y = Dense(1, activation='relu', name='dense_'+str(layer_count))(x_0)
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
    if epoch <= 50:
        lr = initial_lr
    elif epoch <= 100:
        lr = initial_lr/10
    elif epoch <= 200:
        lr = initial_lr/20
    else:
        lr = initial_lr/20
    log('current learning rate is %2.8f' % lr)
    return lr


def train_datagen(epoch_iter=2000, epoch_num=16, batch_size=8, data_dir=args.train_data):
    while(True):
        n_count = 0
        if n_count == 0:
            # print(n_count)
            xs, ys = dg.datag_enerator(data_dir)
            assert len(xs) % args.batch_size == 0, \
                log('make sure the last iteration has a full batchsize, this is important for BN!')
            xs = xs.astype('float32')
            xs = xs/255
            indices = list(range(xs.shape[0]))
            n_count = 1
        for _ in range(epoch_num):
            np.random.shuffle(indices)    # shuffle
            for i in range(0, len(indices), batch_size):
                batch_x = xs[indices[i:i+batch_size]]
                batch_y = ys[indices[i:i+batch_size]]
                yield batch_x, batch_y

# define loss


if __name__ == '__main__':
    # model selection
    AF_model = AFNN(filters=64, image_channels=1, use_bnorm=True)
    AF_model.summary()

    # load the last model in matconvnet style
    initial_epoch = find_LastCheckpoint(model_dir=save_dir)
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        AF_model = load_model(os.path.join(
            save_dir, 'model_%03d.hdf5' % initial_epoch), compile=False)

    # compile the model
    AF_model.compile(optimizer=Adam(0.001),
                     loss=losses.mean_squared_error)

    # use call back functions
    check_pointer = ModelCheckpoint(os.path.join(save_dir, 'model_{epoch:03d}.hdf5'),
                                    verbose=1, save_weights_only=False, period=args.save_step)
    csv_logger = CSVLogger(os.path.join(
        save_dir, 'log.csv'), append=True, separator=',')
    lr_scheduler = LearningRateScheduler(lr_schedule)

    history = AF_model.fit_generator(train_datagen(batch_size=args.batch_size),
                                     steps_per_epoch=40, epochs=args.epoch, verbose=1,
                                     initial_epoch=initial_epoch,
                                     callbacks=[check_pointer, csv_logger, lr_scheduler])
