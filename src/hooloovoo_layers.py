import os
import cPickle as pickle
import cPickle
import cv2
import quick as q
import theano
import sys
import time
import gzip
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import hooloovoo_init as hlv_init
import numpy as np

class ConvLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, activation=T.tanh):
        assert image_shape[1] == filter_shape[1]
        
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])

        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" 
        fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
        
        # initialize weights with random weights
        self.W = hlv_init.standard_init(rng, filter_shape, (fan_in + fan_out))

        # the bias is a 1D tensor -- one bias per output feature map        
        self.b = hlv_init.init_zero((filter_shape[0],))

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        self.output = activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        #self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


class PoolingLayer(object):
    def __init__(self, rng, input, input_shape, poolsize=(2, 2)):
        self.input = input

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=self.input,
                                            ds=poolsize, ignore_border=True)

        # store parameters of this layer
        self.params = []
        
class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = hlv_init.init_zero((n_in,n_out))
        #self.W = theano.shared( theano.shared(value=numpy.zeros((n_in, n_out),
        #                                         dtype=theano.config.floatX),
        #                        name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=np.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self,y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            print self.y_pred, y
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
                     
        self.input = input

        if W is None:
            W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]

