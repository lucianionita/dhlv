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


class ConvLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, activation=T.tanh, poolsize=(2,2)):
        assert image_shape[1] == filter_shape[1]
        
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])

        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" 
        fan_out = filter_shape[0] * np.prod(filter_shape[2:])
        
        # initialize weights with random weights
        self.W = hlv_init.init_conv(rng, filter_shape, poolsize)

        # the bias is a 1D tensor -- one bias per output feature map        
        self.b = hlv_init.init_zero((filter_shape[0],))

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape) # !!! OPTIMIZATable


        self.output = activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

class ConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, activation=T.tanh, poolsize=(2,2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        fan_in = np.prod(filter_shape[1:])
        fan_out = filter_shape[0] * np.prod(filter_shape[2:])
        self.W = hlv_init.init_conv(rng, filter_shape, poolsize)
        self.b = hlv_init.init_zero((filter_shape[0],))


        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))


        # store parameters of this layer
        self.params = [self.W, self.b]   

class FlattenLayer(object):
    def __init__(self, rng, input, data, n_in, n_out):
        self.input = input

        # downsample each feature map individually, using maxpooling
        self.output = self.input.flatten(2)
        
        # store parameters of this layer
        self.params = []

class ReshapeLayer(object):
    def __init__(self, rng, input, data, n_in, n_out):
        self.input = input

        # downsample each feature map individually, using maxpooling
        self.output = self.input.reshape((-1, 1, 28, 28))
        
        # store parameters of this layer
        self.params = []

            
class PoolingLayer(object):
    def __init__(self, rng, input, input_shape, poolsize=(2, 2)):
        self.input = input

        # downsample each feature map individually, using maxpooling
        self.output = downsample.max_pool_2d(input=self.input,
                                            ds=poolsize, ignore_border=True)
        # store parameters of this layer
        self.params = []
    
class DropOutLayer(object):
    def __init__  (self, input, dim):
        self.input = input
        self.dim = dim
        self.drop_mask = hlv_init.init_zero((dim, ))
        self.output = self.drop_mask * self.input
        
class LogisticRegression_dropconnect(object):

    def __init__(self, input, n_in, n_out):

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = hlv_init.init_zero((n_in,n_out))
        #self.W = theano.shared( theano.shared(value=np.zeros((n_in, n_out),
        #                                         dtype=theano.config.floatX),
        #                        name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.drop_mask = hlv_init.init_zero((n_in,n_out))
        self.b = theano.shared(value=np.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W*self.drop_mask) + self.b)

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
            #print "self.ypred:", self.y_pred
            #print "y:", y
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = hlv_init.init_zero((n_in,n_out))
        #self.W = theano.shared( theano.shared(value=np.zeros((n_in, n_out),
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
            #print "self.ypred:", self.y_pred
            #print "y:", y
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation=T.tanh):                     
        self.input = input

        W = hlv_init.init_standard(rng, (n_in, n_out))
        if activation == theano.tensor.nnet.sigmoid:
            W.set_value(W.get_value()*4)

        b = hlv_init.init_zero((n_out,))

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        
        self.params = [self.W, self.b]
        
        self.cost = 0

class HiddenLayer_decomp(object):
    def __init__(self, rng, input, n_in, n_out,
                 activation=T.tanh, length = 10):
        self.input = input

        W_l = hlv_init.init_standard(rng, (n_in, length))
        W_r = hlv_init.init_standard(rng, (length, n_out))

        b = hlv_init.init_zero((n_out,))

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        
        self.params = [self.W, self.b]
        
        self.cost = 0
"""
Layer class
    
"""