# -*- coding: utf-8 -*-

# by Dr. Ming Yan (10/2018)
# yan.meen@gmail.com
# 
# modified on the code from https://github.com/cszn
# =============================================================================

# run this to train the model

# =============================================================================
# For batch normalization layer, momentum should be a value from [0, 0.9] rather than the default 0.99. 
# =============================================================================

import argparse
import re
import os, glob, datetime
import numpy as np
from keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract, MaxPooling2D, Concatenate, Flatten, Dropout
from keras.models import Model, load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
import data_generator as dg
import keras.backend as K

## Params
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='AFNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=10, type=int, help='batch size')
parser.add_argument('--train_data', default='data/Train256', type=str, help='path of train data')
parser.add_argument('--epoch', default=300, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_step', default=1, type=int, help='save model at every x epoches')
args = parser.parse_args()


save_dir = os.path.join('models',args.model+'_'+'256') 

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# Network for focal plane calculation using specialized light path
# training with only few image stacks
# source image stack is 256x256, 200 focal planes with 1um step
# Network use conv layers with Relu and MaxPooling constract features
# use dense layers, dropout layer with softmax as output layes 
# first Conv2D layer:
#	64 filters
#	[3x3], stride 2x2
#	ReLU output [128x128] =================   to path2
#	MaxPooling(2x2) output [64x64]
#	[3x3], stride 2x2
#	ReLU output [32x32]
#	MaxPooling(2x2) [16x16]
#	[3x3], stride 2x2
#	ReLU output [8x8]
#	MaxPooling(2x2)
#	output [4x4] ================   to merge layer
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
#=========================================================================
# solution 1:
# Dense Layer constract the features into 2K
#	ReLU
#	dropout(0.5)
# Dense Layer constract into 2K with softmax
#=========================================================================
# solution 2:
# Dense Layer constract the features into 64
#	ReLU
#	dropout(0.5)
# Dense Layer constract into 1
# use L2 loss

def AFNN(filters=64,image_channels=1, use_bnorm=True):
    layer_count = 0
    inpt = Input(shape=(None,None,image_channels),name = 'input'+str(layer_count))
    # 1st layer, Conv+relu
    layer_count += 1
    x_0 = Conv2D(filters=filters, kernel_size=(3,3), strides=(2,2),kernel_initializer='Orthogonal', padding='same',name = 'conv'+str(layer_count))(inpt)
    layer_count += 1
    x_0 = Activation('relu',name = 'relu'+str(layer_count))(x_0)
	# Path 1
	layer_count += 1
    x1 = MaxPooling2D(pool_size=(2,2), strides=None, padding='same', data_format=None, name = 'maxpool'+str(layer_count))(x_0)
    # 2 layers, Conv+BN+relu+MaxPooling
    for i in range(2):
        layer_count += 1
        x1 = Conv2D(filters=filters, kernel_size=(3,3), strides=(2,2),kernel_initializer='Orthogonal', padding='same',use_bias = False,name = 'conv'+str(layer_count))(x1)
        if use_bnorm:
            layer_count += 1
            x1 = BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'bn'+str(layer_count))(x1)
        layer_count += 1
        x1 = Activation('relu',name = 'relu'+str(layer_count))(x1) 
		x1 = MaxPooling2D(pool_size=(2,2), strides=None, padding='same', data_format=None, name = 'maxpool'+str(layer_count))(x1)
		
	# Path 2
	# 2 layers, Conv+BN+ReLU+MaxPooling	
	layer_count += 1
    x_1 = Conv2D(filters=filters, kernel_size=(7,7), strides=(2,2),kernel_initializer='Orthogonal', padding='same',name = 'conv'+str(layer_count))(x_0)
    layer_count += 1
    x_1 = Activation('relu',name = 'relu'+str(layer_count))(x_1)
	
	# Path 2_1
	# 1 layer, Conv+BN+ReLU+MaxPooling
	layer_count += 1
    x2 = MaxPooling2D(pool_size=(2,2), strides=None, padding='same', data_format=None, name = 'maxpool'+str(layer_count))(x_1)
    layer_count += 1
	x2 = Conv2D(filters=filters, kernel_size=(5,5), strides=(2,2),kernel_initializer='Orthogonal', padding='same',use_bias = False,name = 'conv'+str(layer_count))(x2)
	if use_bnorm:
		layer_count += 1
		x2 = BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'bn'+str(layer_count))(x2)
	layer_count += 1
	x2 = Activation('relu',name = 'relu'+str(layer_count))(x2) 
	x2 = MaxPooling2D(pool_size=(2,2), strides=None, padding='same', data_format=None, name = 'maxpool'+str(layer_count))(x2)
	
	# Path 2_2
	# 1 layer, Conv+BN+ReLU+MaxPooling
	layer_count += 1
    x_2 = Conv2D(filters=filters, kernel_size=(7,7), strides=(4,4),kernel_initializer='Orthogonal', padding='same',name = 'conv'+str(layer_count))(x_1)
    if use_bnorm:
		layer_count += 1
		x3 = BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'bn'+str(layer_count))(x_2)
	layer_count += 1
    x3 = Activation('relu', name = 'relu'+str(layer_count))(x3)
	layer_count += 1
    x3 = MaxPooling2D(pool_size=(2,2), strides=None, padding='same', data_format=None, name = 'maxpool'+str(layer_count))(x3)
	
    # Merge layer
	layer_count += 1
    x = Concatenate(axis=-1, name = 'concat'+str(layer_count))([x1,x2,x3])
	layer_count += 1
	x = Flatten(data_format=None, name = 'Flat'+str(layer_count))(x)
	
	# Dense layer
    layer_count += 1
    x = Dense(2048, activation='relu', name = 'dense'+str(layer_count))(x)
	layer_count += 1
    x = Dropout(0.5, name = 'dropout'+str(layer_count))(x)
    layer_count += 1
    x = Dense(2048, activation='softmax', name = 'dense'+str(layer_count))(x)
    model = Model(inputs=inpt, outputs=x)
    
    return model


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir,'model_*.hdf5'))  # get name list of all .hdf5 files
    #file_list = os.listdir(save_dir)
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).hdf5.*",file_)
            #print(result[0])
            epochs_exist.append(int(result[0]))
        initial_epoch=max(epochs_exist)   
    else:
        initial_epoch = 0
    return initial_epoch

def log(*args,**kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"),*args,**kwargs)

def lr_schedule(epoch):
    initial_lr = args.lr
    if epoch<=30:
        lr = initial_lr
    elif epoch<=60:
        lr = initial_lr/10
    elif epoch<=80:
        lr = initial_lr/20 
    else:
        lr = initial_lr/20 
    log('current learning rate is %2.8f' %lr)
    return lr

def train_datagen(epoch_iter=2000,epoch_num=5,batch_size=128,data_dir=args.train_data):
    while(True):
        n_count = 0
        if n_count == 0:
            #print(n_count)
            xs = dg.datagenerator(data_dir)
            assert len(xs)%args.batch_size ==0, \
            log('make sure the last iteration has a full batchsize, this is important if you use batch normalization!')
            xs = xs.astype('float32')
            indices = list(range(xs.shape[0]))
            n_count = 1
        for _ in range(epoch_num):
            np.random.shuffle(indices)    # shuffle
            for i in range(0, len(indices), batch_size):
                batch_x = xs[indices[i:i+batch_size]]
                #noise =  np.random.normal(0, args.sigma/255.0, batch_x.shape)    # noise
                #noise =  K.random_normal(ge_batch_y.shape, mean=0, stddev=args.sigma/255.0)
                batch_y = batch_x + noise 
                yield batch_y, batch_x
        
# define loss
def sum_squared_error(y_true, y_pred):
    #return K.mean(K.square(y_pred - y_true), axis=-1)
    #return K.sum(K.square(y_pred - y_true), axis=-1)/2
    return K.sum(K.square(y_pred - y_true))/2
    
if __name__ == '__main__':
    # model selection
    model = AFNN(depth=5,filters=64,image_channels=1,use_bnorm=True)
    model.summary()
    
    # load the last model in matconvnet style
    initial_epoch = findLastCheckpoint(save_dir=save_dir)
    if initial_epoch > 0: 
        print('resuming by loading epoch %03d'%initial_epoch)
        model = load_model(os.path.join(save_dir,'model_%03d.hdf5'%initial_epoch), compile=False)
    
    # compile the model
    model.compile(optimizer=Adam(0.001), loss=sum_squared_error)
    
    # use call back functions
    checkpointer = ModelCheckpoint(os.path.join(save_dir,'model_{epoch:03d}.hdf5'), 
                verbose=1, save_weights_only=False, period=args.save_every)
    csv_logger = CSVLogger(os.path.join(save_dir,'log.csv'), append=True, separator=',')
    lr_scheduler = LearningRateScheduler(lr_schedule)
    
    history = model.fit_generator(train_datagen(batch_size=args.batch_size),
                steps_per_epoch=2000, epochs=args.epoch, verbose=1, initial_epoch=initial_epoch,
                callbacks=[checkpointer,csv_logger,lr_scheduler])
				