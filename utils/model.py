
"""
Sen2LCZ
Reference: https://ieeexplore.ieee.org/document/9103196
Multilevel Feature Fusion-Based CNN for Local Climate Zone Classification From Sentinel-2 Images: Benchmark Results on the So2Sat LCZ42 Dataset
Source: https://github.com/ChunpingQiu/benchmark-on-So2SatLCZ42-dataset-a-simple-tour
"""

from __future__ import print_function
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
import numpy as np


def sen2LCZ_drop_core(inputs, num_classes=17, bn=1, depth=5, dim=16, dropRate=0.1, fusion=0):

    # Start model definition.
    inc_rate = 2

    lay_per_block=int((depth-1)/4)

    conv0 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(inputs)

    if bn==1:
        conv0 = BatchNormalization(axis=-1)(conv0)
    conv0 = Activation('relu')(conv0)

    for i in np.arange(lay_per_block-1):
        conv0 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(conv0)
        if bn==1:
            conv0 = BatchNormalization(axis=-1)(conv0)
        conv0 = Activation('relu')(conv0)

    pool0 = MaxPooling2D((2, 2))(conv0)
    pool1 = AveragePooling2D(pool_size=2)(conv0)
    merge0 = Concatenate()([pool0,pool1])

    if fusion==1:
        'prediction'
        x = GlobalAveragePooling2D()(merge0)#Flatten
        outputs_32 = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(x)

    dim=dim*inc_rate
    conv1 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(merge0)
    if bn==1:
        conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = Activation('relu')(conv1)

    for i in np.arange(lay_per_block-1):
        conv1 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(conv1)
        if bn==1:
            conv1 = BatchNormalization(axis=-1)(conv1)
        conv1 = Activation('relu')(conv1)

    pool0 = MaxPooling2D((2, 2))(conv1)
    pool1 = AveragePooling2D(pool_size=2)(conv1)
    merge1 = Concatenate()([pool0,pool1])

    'dropOut'
    merge1 = Dropout(dropRate)(merge1)

    if fusion==1:
        'prediction'
        x = GlobalAveragePooling2D()(merge1)#Flatten
        outputs_16 = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(x)

    dim=dim*inc_rate
    conv2 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(merge1)
    if bn==1:
        conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = Activation('relu')(conv2)

    for i in np.arange(lay_per_block-1):
        conv2 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(conv2)
        if bn==1:
            conv2 = BatchNormalization(axis=-1)(conv2)
        conv2 = Activation('relu')(conv2)

    pool0 = MaxPooling2D((2, 2))(conv2)
    pool1 = AveragePooling2D(pool_size=2)(conv2)
    merge2 = Concatenate()([pool0,pool1])

    'dropOut'
    merge2 = Dropout(dropRate)(merge2)

    if fusion==1:
        'prediction'
        x = GlobalAveragePooling2D()(merge2)#Flatten
        outputs_8 = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(x)

    dim=dim*inc_rate
    conv3 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(merge2)
    if bn==1:
        conv3 = BatchNormalization(axis=-1)(conv3)
    conv3 = Activation('relu')(conv3)

    for i in np.arange(lay_per_block-1):
        conv3 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(conv3)
        if bn==1:
            conv3 = BatchNormalization(axis=-1)(conv3)
        conv3 = Activation('relu')(conv3)

    'prediction'
    x = GlobalAveragePooling2D()(conv3)#Flatten
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    if fusion==1:
        'final prediction'
        o=outputs=Average()([outputs, outputs_32, outputs_16, outputs_8])
    else:
        o=outputs

    return o

def sen2LCZ_drop(input_shape=(32,32,10), num_classes=17, bn=1, depth=5, dim=16, dropRate=0.1, fusion=0):
    """

    # Arguments
        input_shape: Array with input dimensions
        ...
    # Returns
        model (Model): Keras model instance
    """

    inputs = Input(shape=input_shape)
    o=sen2LCZ_drop_core(inputs, num_classes=num_classes, bn=bn, depth=depth, dim=dim, dropRate=dropRate, fusion=fusion)

    return Model(inputs=inputs, outputs=o)
