import math

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import DepthwiseConv2D, ZeroPadding2D, Conv2D, PReLU, BatchNormalization
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from model_mobilenet.arcface_func import *
# from arcface_func import *
# weight decay setting
weight_decay = 4e-5


def bottleneck(x_in, d_in, d_out, stride, depth_multiplier):
    # decide whether there would be a short cut
    if stride == 1 and d_in == d_out:
        connect = True
    else:
        connect = False

    # point-wise layers
    x = Conv2D(d_in * depth_multiplier, kernel_size=1, strides=1, padding='VALID',
               use_bias=False, kernel_regularizer=l2(weight_decay))(x_in)
    x = BatchNormalization()(x)
    x = PReLU(shared_axes=[1, 2])(x)

    # depth-wise layers
    x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)  # manually padding
    x = keras.layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding='VALID',
                                     use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.PReLU(shared_axes=[1, 2])(x)

    # point-wise layers linear
    x = keras.layers.Conv2D(d_out, kernel_size=1, strides=1, padding='VALID',
                            use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    x = keras.layers.BatchNormalization()(x)

    if connect:
        return keras.layers.Add()([x, x_in])
    else:
        return x


def conv_block(x_in, d_in, d_out, kernel_size, stride, padding, depthwise=False, linear=False):
    # padding if needed
    if padding != 0:
        x = ZeroPadding2D(padding=(padding, padding))(x_in)
    else:
        x = x_in

    if depthwise:
        x = DepthwiseConv2D(kernel_size, strides=stride, padding='VALID',
                            use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    else:
        x = Conv2D(d_out, kernel_size, strides=stride, padding='VALID',
                   use_bias=False, kernel_regularizer=l2(weight_decay))(x)

    x = BatchNormalization()(x)

    if not linear:
        return PReLU(shared_axes=[1, 2])(x)
    else:
        return x


Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]


def Mobilefacenet_Arcface(input_shape=None, inplanes=64,
                  classes=1000,
                  num_feature=128,
                  setting=Mobilefacenet_bottleneck_setting):
    input = Input(shape=input_shape)
    y = Input(shape=(classes,))
    x = conv_block(input, d_in=3, d_out=64, kernel_size=3, stride=2, padding=1)
    x = conv_block(x, d_in=64, d_out=64, kernel_size=3, stride=1, padding=1, depthwise=True)
    for t, c, n, s in setting:
        for i in range(n):
            if i == 0:
                x = bottleneck(x, inplanes, c, s, t)
            else:
                x = bottleneck(x, inplanes, c, 1, t)
            inplanes = c
    x = conv_block(x, d_in=128, d_out=512, kernel_size=1, stride=1, padding=0)
    x = conv_block(x, d_in=512, d_out=512, kernel_size=(7, 6), stride=1, padding=0,
                   depthwise=True, linear=True)
    x = conv_block(x, d_in=512, d_out=num_feature, kernel_size=1, stride=1, padding=0,
                   linear=True)
    x = keras.layers.Flatten()(x)
    y = Input(shape=(classes,))
    output = ArcFace_v2(n_classes=classes)((x, y))
    return Model([input, y], output)


if __name__ == '__main__':
    model = Mobilefacenet_Arcface(input_shape=(112, 96, 3), num_feature=128, classes=10572)
    model.summary()
