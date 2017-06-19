#
# This code can be added to switch between multiple size models using this code:
#
# import models
# input_var, input_var_ex, net = models.model(patch_size,framesize, noutputs, stride)
#


import sys,os,time,random
import numpy as np

import theano
import theano.tensor as T 
import lasagne
from lasagne.layers.normalization import BatchNormLayer
from lasagne.layers import InputLayer, ConcatLayer, Conv2DLayer

import pickle
import scipy

def ConvFactory(data, num_filter, filter_size, stride=1, pad=0, nonlinearity=lasagne.nonlinearities.leaky_rectify):
    data = lasagne.layers.batch_norm(Conv2DLayer(
        data, num_filters=num_filter,
        filter_size=filter_size,
        stride=stride, pad=pad,
        nonlinearity=nonlinearity,
        W=lasagne.init.GlorotUniform(gain='relu')))
    return data

def SimpleFactory(data, ch_1x1, ch_3x3):
    conv1x1 = ConvFactory(data=data, filter_size=1, pad=0, num_filter=ch_1x1)
    conv3x3 = ConvFactory(data=data, filter_size=3, pad=1, num_filter=ch_3x3) 
    concat = ConcatLayer([conv1x1, conv3x3])
    return concat


def model(patch_size,framesize, noutputs, stride):
    if patch_size == 32:
        return model_32x32(framesize, noutputs, stride)
    elif patch_size == 64:
        return model_64x64(framesize, noutputs, stride)
    elif patch_size == 128:
        return model_128x128(framesize, noutputs, stride)
    else:
        raise Exception('No network for that size')


def model_32x32(framesize, noutputs, stride):

    patch_size = 32

    input_var = T.tensor4('inputs')
    input_var_ex = T.ivector('input_var_ex')

    input_shape = (None, 1, framesize, framesize)
    img = InputLayer(shape=input_shape, input_var=input_var[input_var_ex])
    net = img

    net = ConvFactory(net, filter_size=3, num_filter=64, pad=patch_size)
    print net.output_shape
    net = SimpleFactory(net, 16, 16)
    print net.output_shape
    net = SimpleFactory(net, 16, 32)
    print net.output_shape
    net = ConvFactory(net, filter_size=14, num_filter=16) 
    print net.output_shape
    net = SimpleFactory(net, 112, 48)
    print net.output_shape
    net = SimpleFactory(net, 64, 32)
    print net.output_shape
    net = SimpleFactory(net, 40, 40)
    print net.output_shape
    net = SimpleFactory(net, 32, 96)
    print net.output_shape
    net = ConvFactory(net, filter_size=20, num_filter=32) 
    print net.output_shape
    net = ConvFactory(net, filter_size=1, pad=0, num_filter=64)
    print net.output_shape
    net = ConvFactory(net, filter_size=1, pad=0, num_filter=64)
    print net.output_shape
    net = ConvFactory(net, filter_size=1, num_filter=1, stride=stride)
    print net.output_shape
    
    return input_var, input_var_ex, net


def model_64x64(framesize, noutputs, stride):

    patch_size = 64
    input_var = T.tensor4('inputs')
    input_var_ex = T.ivector('input_var_ex')

    input_shape = (None, 1, framesize, framesize)
    img = InputLayer(shape=input_shape, input_var=input_var[input_var_ex])
    net = img

    net = ConvFactory(net, filter_size=3, num_filter=64, pad=patch_size)
    print net.output_shape
    net = SimpleFactory(net, 16, 16)
    print net.output_shape
    net = SimpleFactory(net, 16, 32)
    print net.output_shape
    net = ConvFactory(net, filter_size=24, num_filter=16) 
    print net.output_shape
    net = SimpleFactory(net, 112, 48)
    print net.output_shape
    net = SimpleFactory(net, 64, 32)
    print net.output_shape
    net = SimpleFactory(net, 40, 40)
    print net.output_shape
    net = SimpleFactory(net, 32, 96)
    print net.output_shape
    net = ConvFactory(net, filter_size=42, num_filter=32) 
    print net.output_shape
    net = ConvFactory(net, filter_size=1, pad=0, num_filter=64)
    print net.output_shape
    net = ConvFactory(net, filter_size=1, pad=0, num_filter=64)
    print net.output_shape
    net = ConvFactory(net, filter_size=1, num_filter=1, stride=stride)
    print net.output_shape
    
    return input_var, input_var_ex, net



def model_128x128(framesize, noutputs, stride):

    patch_size =128
    input_var = T.tensor4('inputs')
    input_var_ex = T.ivector('input_var_ex')

    input_shape = (None, 1, framesize, framesize)
    img = InputLayer(shape=input_shape, input_var=input_var[input_var_ex])
    net = img

    net = ConvFactory(net, filter_size=3, num_filter=64, pad=patch_size)
    print net.output_shape
    net = SimpleFactory(net, 16, 16)
    print net.output_shape
    net = SimpleFactory(net, 16, 32)
    print net.output_shape
    net = ConvFactory(net, filter_size=64, num_filter=16) 
    print net.output_shape
    net = SimpleFactory(net, 112, 48)
    print net.output_shape
    net = SimpleFactory(net, 64, 32)
    print net.output_shape
    net = SimpleFactory(net, 40, 40)
    print net.output_shape
    net = SimpleFactory(net, 32, 96)
    print net.output_shape
    net = ConvFactory(net, filter_size=66, num_filter=32) 
    print net.output_shape
    net = ConvFactory(net, filter_size=1, pad=0, num_filter=64)
    print net.output_shape
    net = ConvFactory(net, filter_size=1, pad=0, num_filter=64)
    print net.output_shape
    net = ConvFactory(net, filter_size=1, num_filter=1, stride=stride)
    print net.output_shape
    
    return input_var, input_var_ex, net


