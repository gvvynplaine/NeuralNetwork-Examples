"""
simplified inception-bn.py for images has size around 15 x 15
"""

import mxnet as mx

def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), act_type="relu"):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
    #bn = mx.symbol.BatchNorm(data=conv)
    act = mx.symbol.Activation(data = conv, act_type=act_type)
    return act

def get_symbol(num_classes = 2, num_filter=20):
    data = mx.symbol.Variable(name="data")
    #data = data/255
    conv1 = ConvFactory(data=data, kernel=(4,4), pad=(1,1), num_filter=num_filter, act_type="relu")
    conv2 = ConvFactory(data=conv1, kernel=(4,4), pad=(1,1), num_filter=num_filter, act_type="relu")
    flatten = mx.symbol.Flatten(data=conv2, name="flatten1")
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500, name="fc1")
    fc2 = mx.symbol.FullyConnected(data=fc1, num_hidden=num_classes, name="fc2")
    softmax = mx.symbol.SoftmaxOutput(data=fc2, name="softmax")
    return softmax
