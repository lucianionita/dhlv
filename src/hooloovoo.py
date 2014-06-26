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
import time
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from hooloovoo_aux import mnist_data, faces_data
import hooloovoo_aux as hlv_aux
import hooloovoo_init as hlv_init
import hooloovoo_layers as hlv_layers
import hooloovoo_tests
import hooloovoo_train as hlv_train
import hooloovoo_models as hlv_models


rectifier = lambda x: T.maximum(0, x)

def show(x, r=1, size=64):
    q.show((x.reshape(size,size)-np.min(x)) / (np.max(x) - np.min(x)), r=r)

def show2(x, r=1, size=64):
    q.show((x.reshape(size,size)) / (np.max(np.abs(x)))+0.5, r=r)

def showAll(X, r=1, size=64):
    for i in range (X.shape[1]):
        show(param_values[0][:,i], r, size)

def showAll2(X, r=1, size=64):
    for i in range (X.shape[1]):
        show2(param_values[0][:,i], r, size)

def showBunch(X, r=1, size=64):
    n = X.size/size/size
    s = size
    for i in range(732,733):
        Y = X.copy()
        if i & 16: Y = Y.reshape(s,s,n)
        if i & 64: Y = Y.transpose()
        if i & 128: Y = Y.reshape(n*s,s)
        if i & 512: Y = Y.transpose()
        Y = Y.reshape(s,s*n)        
        print i
        q.show(Y/(np.max(Y))+0.5, r=r, t=-1)




if 'data_faces' in dir(): 
    print "... Data seems to be already loaded"
else:
    data_faces = faces_data()
    """
if 'data_digits' in dir(): 
    print "... Data seems to be already loaded"
else:
    data_digits = mnist_data()
"""

"""
lr = 0.05
lr_exp = 0.5
batch_size = 200
n_hidden = 100
MLP2 = hlv_models.MLP_model(64*64, 2, data_faces, batch_size, None, n_hidden, 1)
param_values = hlv_aux.get_params(MLP2)
while lr > 0.00001:
    print "============ LR %f" % (lr)
    MLP2 = hlv_models.MLP_model(64*64, 2, data_faces, batch_size, None, n_hidden, lr)
    hlv_aux.set_params(MLP2, param_values)
    param_values = hlv_train.train_minibatches(MLP2, max_epochs=1000)
    lr = lr * lr_exp
    
"""

"""
lr = 0.1
lr_exp = 0.5
batch_size = 500
prob_drop = 0.333
n_hidden = 100
M = hlv_models.MLP_model_dropout2(64*64, 2, data_faces, batch_size, None, n_hidden, 1, prob_drop=prob_drop)

#M = hlv_models.MLP_model(64*64, 2, data_faces, batch_size, None, n_hidden, 1)
param_values = hlv_aux.get_params(M)
while lr > 0.00001:
    print "============ LR %f" % (lr)
    M = hlv_models.MLP_model_dropout2(64*64, 2, data_faces, batch_size, None, n_hidden, lr, prob_drop=prob_drop)
    #M = hlv_models.MLP_model(64*64, 2, data_faces, batch_size, None, n_hidden, lr)
    hlv_aux.set_params(M, param_values)
    param_values = hlv_train.train_minibatches(M, max_epochs=100)
    lr = lr * lr_exp
"""



reload(hlv_models)
reload(hlv_train)




"""
Layerspec structure:
    (layertype, paramDict)
    

Layer specs:
    Must have as init params:
        input
        n_in
        n_out
        data
        rng
        activation

    Must expose:
        output    
        params
        
    
"""
class CNN_C5S2C5S2H500L_model():
    def __init__(self, n_in, n_out, data, n_hidden, batch_size, rng, learning_rate, activation=T.tanh, L1_reg=0., L2_reg=0.0001):

        # Define classifier independent stuff
        #####################################

        index = T.lscalar()  # index to a [mini]batch
        x = T.matrix('x')   
        y = T.ivector('y')  
        X = T.matrix('X')  
        Y = T.ivector('Y')                             
        
        if (rng is None):
            rng = np.random.RandomState(23455)        
        self.batch_size = batch_size
        self.n_train_batches = data.train.x.get_value(borrow=True).shape[0] / batch_size
        self.n_valid_batches = data.valid.x.get_value(borrow=True).shape[0] / batch_size
        self.n_test_batches  = data.test.x.get_value (borrow=True).shape[0] / batch_size        
        
        self.x = x        
        self.y = y
        self.data = data 
        
        
        # Define the model structure    
        ############################
        """
        layerClasses = {
            'conv': hlv_layers.ConvLayer,
            'pool': hlv_layers.PoolingLayer,
            'hidden': hlv_layers.HiddenLayer,
            'logistic': hlv_layers.LogisticRegression,
            'flatten': hlv_layers.FlattenLayer
            'reshape': hlv_layers.ReshapeLayer
            }
        
        Layers = []
        layer_input = x
        layer_n_input = n_in
        for layer_idx in range(len(layerSpecs)):
            layerType, layerConfig = layerSpecs[layer_idx]            
            layerClass = layerClasses[layerType]
            new_layer = layerClass(        input = layer_input,
                                           shape_in = layer_n_in,                                           
                                           data = data,
                                           rng = rng, 
                                           **config)
        """    
            
        self.convLayer1 = hlv_layers.ConvLayer( rng = rng, 
                                                input = x.reshape((-1, 1, 28, 28)), 
                                                filter_shape = (n_hidden[0], 1, 5, 5), 
                                                image_shape = (batch_size, 1, 28, 28), 
                                                activation=activation,
                                                poolsize=(2, 2))
        self.poolingLayer1 = hlv_layers.PoolingLayer(rng = rng, 
                                                     input = self.convLayer1.output,
                                                     input_shape=(0),
                                                     poolsize=(2, 2))
        self.convLayer2 = hlv_layers.ConvLayer( rng = rng, 
                                                input = self.poolingLayer1.output.reshape((-1, n_hidden[0], 12, 12)), 
                                                filter_shape = (n_hidden[1], n_hidden[0], 5, 5), 
                                                image_shape = (batch_size, n_hidden[0], 12, 12), 
                                                activation=activation,
                                                poolsize=(2, 2))
        self.poolingLayer2 = hlv_layers.PoolingLayer(rng = rng, 
                                                     input = self.convLayer2.output,
                                                     input_shape=(0),
                                                     poolsize=(2, 2))
        self.hiddenLayer = hlv_layers.HiddenLayer(   rng=rng,
                                                    input = self.poolingLayer2.output.flatten(2), 
                                                    n_in = n_hidden[1] * 4 * 4,
                                                    n_out = 500, 
                                                    activation=activation)
#        self.logRegressionLayer = hlv_layers.LogisticRegression(
#                                    input=self.hiddenLayer.output,
#                                    n_in=500,
#                                    n_out=n_out)
        self.logRegressionLayer = hlv_layers.LogisticRegression(
                                    input=self.hiddenLayer.output,
                                    n_in=500,
                                    n_out=n_out)
        
        


        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.errors = self.logRegressionLayer.errors

        # define the cost
        self.cost = self.negative_log_likelihood(y)
            #\
            # + L1_reg * self.L1 \
            # + L2_reg * self.L2_sqr
    
        # define parameters
        self.params = self.convLayer1.params + self.convLayer2.params + self.hiddenLayer.params+ self.logRegressionLayer.params
        #self.params = self.logRegressionLayer.params
 
        self.grads = [T.grad(cost=self.cost, wrt=param) for param in self.params]
        
        self.updates = [(param, param - learning_rate * grad) for
                    param, grad in zip(self.params, self.grads)]
    
        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`

        """
        self.train_model = theano.function(inputs=[X, Y],
                                            outputs=self.errors(y),
                                            updates=self.updates,
                                            givens={
                                                x: X,
                                                y: Y})
        self.test_model = theano.function(  inputs=[],
                                            outputs=self.errors(y),
                                            givens={
                                                x: self.data.test.x,
                                                y: self.data.test.y})
                                                
        self.validate_model = theano.function(inputs=[],
                                            outputs=self.errors(y),
                                            givens={
                                                x: self.data.valid.x,
                                                y: self.data.valid.y}) 
        """                                                
                                                
        self.test_model_minibatch = theano.function(  inputs=[index],
                                            outputs=self.errors(y),
                                            givens={
                                                x: self.data.test.x[index * self.batch_size:(index + 1) * self.batch_size],
                                                y: self.data.test.y[index * self.batch_size:(index + 1) * self.batch_size]})
                                                
        self.validate_model_minibatch = theano.function(inputs=[index],
                                            outputs=self.errors(y),
                                            givens={
                                                x: self.data.valid.x[index * self.batch_size:(index + 1) * self.batch_size],
                                                y: self.data.valid.y[index * self.batch_size:(index + 1) * self.batch_size]}) 
        self.train_model_minibatch = theano.function(inputs=[index],
                                            outputs=self.cost,
                                            updates=self.updates,
                                            givens={
                                                x: self.data.train.x[index * self.batch_size:(index + 1) * self.batch_size],
                                                y: self.data.train.y[index * self.batch_size:(index + 1) * self.batch_size]})                  
    def validate_model(self):
        validation_losses = [self.validate_model_minibatch(i) for i
                                     in xrange(self.n_valid_batches)]
        return np.mean(validation_losses)
    def test_model(self):
        test_losses = [self.test_model_minibatch(i) for i
                                     in xrange(self.n_test_batches)]
        return np.mean(test_losses)    
    
class CNN_faces_model():
    def __init__(self, n_in, n_out, data, n_hidden, batch_size, rng, learning_rate, activation=T.tanh, L1_reg=0., L2_reg=0.0001):

        # Define classifier independent stuff
        #####################################

        index = T.lscalar()  # index to a [mini]batch
        x = T.matrix('x')   
        y = T.ivector('y')  
        X = T.matrix('X')  
        Y = T.ivector('Y')                             
        
        if (rng is None):
            rng = np.random.RandomState(23455)        
        self.batch_size = batch_size
        self.n_train_batches = data.train.x.get_value(borrow=True).shape[0] / batch_size
        self.n_valid_batches = data.valid.x.get_value(borrow=True).shape[0] / batch_size
        self.n_test_batches  = data.test.x.get_value (borrow=True).shape[0] / batch_size        
        
        self.x = x        
        self.y = y
        self.data = data 
        
        
        # Define the model structure    
        ############################
        """
        layerClasses = {
            'conv': hlv_layers.ConvLayer,
            'pool': hlv_layers.PoolingLayer,
            'hidden': hlv_layers.HiddenLayer,
            'logistic': hlv_layers.LogisticRegression,
            'flatten': hlv_layers.FlattenLayer
            'reshape': hlv_layers.ReshapeLayer
            }
        
        Layers = []
        layer_input = x
        layer_n_input = n_in
        for layer_idx in range(len(layerSpecs)):
            layerType, layerConfig = layerSpecs[layer_idx]            
            layerClass = layerClasses[layerType]
            new_layer = layerClass(        input = layer_input,
                                           shape_in = layer_n_in,                                           
                                           data = data,
                                           rng = rng, 
                                           **config)
        """    
            
        self.convLayer1 = hlv_layers.ConvLayer( rng = rng, 
                                                input = x.reshape((-1, 1, 64, 64)), 
                                                filter_shape = (n_hidden[0], 1, 13, 13), 
                                                image_shape = (batch_size, 1, 64, 64), 
                                                activation=activation,
                                                poolsize=(4, 4))
        self.poolingLayer1 = hlv_layers.PoolingLayer(rng = rng, 
                                                     input = self.convLayer1.output,
                                                     input_shape=(0),
                                                     poolsize=(4, 4))
        self.convLayer2 = hlv_layers.ConvLayer( rng = rng, 
                                                input = self.poolingLayer1.output.reshape((-1, n_hidden[0], 13, 13)), 
                                                filter_shape = (n_hidden[1], n_hidden[0], 4, 4), 
                                                image_shape = (batch_size, n_hidden[0], 13, 13), 
                                                activation=activation,
                                                poolsize=(2, 2))
        self.poolingLayer2 = hlv_layers.PoolingLayer(rng = rng, 
                                                     input = self.convLayer2.output,
                                                     input_shape=(0),
                                                     poolsize=(2, 2))
        self.hiddenLayer = hlv_layers.HiddenLayer(   rng=rng,
                                                    input = self.poolingLayer2.output.flatten(2), 
                                                    n_in = n_hidden[1] * 5 * 5,
                                                    n_out = 500, 
                                                    activation=activation)

        self.logRegressionLayer = hlv_layers.LogisticRegression(
                                    input=self.hiddenLayer.output,
                                    n_in=500,
                                    n_out=n_out)
        
        


        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.errors = self.logRegressionLayer.errors

        # define the cost
        self.cost = self.negative_log_likelihood(y)
            #\
            # + L1_reg * self.L1 \
            # + L2_reg * self.L2_sqr
    
        # define parameters
        self.params = self.convLayer1.params + self.convLayer2.params + self.hiddenLayer.params+ self.logRegressionLayer.params
        #self.params = self.logRegressionLayer.params
 
        self.grads = [T.grad(cost=self.cost, wrt=param) for param in self.params]
        
        self.updates = [(param, param - learning_rate * grad) for
                    param, grad in zip(self.params, self.grads)]
    
        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`

        """
        self.train_model = theano.function(inputs=[X, Y],
                                            outputs=self.errors(y),
                                            updates=self.updates,
                                            givens={
                                                x: X,
                                                y: Y})
        self.test_model = theano.function(  inputs=[],
                                            outputs=self.errors(y),
                                            givens={
                                                x: self.data.test.x,
                                                y: self.data.test.y})
                                                
        self.validate_model = theano.function(inputs=[],
                                            outputs=self.errors(y),
                                            givens={
                                                x: self.data.valid.x,
                                                y: self.data.valid.y}) 
        """                                                
                                                
        self.test_model_minibatch = theano.function(  inputs=[index],
                                            outputs=self.errors(y),
                                            givens={
                                                x: self.data.test.x[index * self.batch_size:(index + 1) * self.batch_size],
                                                y: self.data.test.y[index * self.batch_size:(index + 1) * self.batch_size]})
                                                
        self.validate_model_minibatch = theano.function(inputs=[index],
                                            outputs=self.errors(y),
                                            givens={
                                                x: self.data.valid.x[index * self.batch_size:(index + 1) * self.batch_size],
                                                y: self.data.valid.y[index * self.batch_size:(index + 1) * self.batch_size]}) 
        self.train_model_minibatch = theano.function(inputs=[index],
                                            outputs=self.cost,
                                            updates=self.updates,
                                            givens={
                                                x: self.data.train.x[index * self.batch_size:(index + 1) * self.batch_size],
                                                y: self.data.train.y[index * self.batch_size:(index + 1) * self.batch_size]})                  
    def validate_model(self):
        validation_losses = [self.validate_model_minibatch(i) for i
                                     in xrange(self.n_valid_batches)]
        return np.mean(validation_losses)
    def test_model(self):
        test_losses = [self.test_model_minibatch(i) for i
                                     in xrange(self.n_test_batches)]
        return np.mean(test_losses)    


class GNN():
    def __init__():
        print "There are several issues here."
        print " First, the GenericNN should receive as input ONE parameter, "
        print "     which is the config structure that holds EVERYTHING."
        print " Secondly, this means that the individual layers get config"
        print "     structures. So they must also handle this."
        print " Thirdly, the model should have different handles for (training,"
        print "     testing, validating) batch, whole, individual, arbitray."
        print "     It would be best to just compile these as needed, on the"
        print "     fly, in a memoization-type of way. Also, they would be defined"
        print "     So we need a generic function that ads these methods to the "
        print "     individual models."
        print " Fourthly, we want to make the system as simple and elegant to use"
        print "     as possible, which is a goal at ods with our flexibility wishes."
        print " Fifthly, we want to have generic methods of the layers which tell "
        print "     us stuff about them, like param size, output shape etc."
        print " Sixthly, I should add different datasets like mnist, faces. Also"
        print "     some preprocessing is in order, like contrast normalization,"
        print "     rescaling, whitening."
        print " Seventhly, data augmentation should be included: noise, jitter, "
        print "     warp, distort, mirror, etc."
        print " Eightly, it should be possible to configure experiments to run."
        print "     And also record ALL the possible data."
        print " Ninthly, there should be better learning algorithms."
        print " Tenthly, and finally, all these changes should be accompanied by"
        print "     a new object/file structure, which warrants a rewrite to a"
        print "     *voo* module.
        
        pass

reload(hlv_layers)
reload(hlv_models)
reload(hlv_train)
lr = 0.0001
lr_exp = 0.5
batch_size = 500
hid = [20, 50]
"""
M = CNN_C5S2C5S2H500L_model(28*28, 10, data_digits, hid, batch_size, None, lr)
param_values = hlv_aux.get_params(M)
while lr > 0.00001:
    print "============ LR %f" % (lr)
    M = CNN_C5S2C5S2H500L_model(28*28, 10, data_digits, hid, batch_size, None, lr)
    #M = hlv_models.MLP_model(64*64, 2, data_faces, batch_size, None, n_hidden, lr)
    hlv_aux.set_params(M, param_values)
    param_values = hlv_train.train_minibatches(M, min_epochs=50, max_epochs=1000)
    lr = lr * lr_exp

"""

M = CNN_faces_model(64*64, 10, data_faces, hid, batch_size, None, lr)
param_values = hlv_aux.get_params(M)
while lr > 0.00001:
    print "============ LR %f" % (lr)
    M = CNN_faces_model(64*64, 10, data_faces, hid, batch_size, None, lr)
    #M = hlv_models.MLP_model(64*64, 2, data_faces, batch_size, None, n_hidden, lr)
    hlv_aux.set_params(M, param_values)
    param_values = hlv_train.train_minibatches(M, min_epochs=20, max_epochs=200)
    lr = lr * lr_exp