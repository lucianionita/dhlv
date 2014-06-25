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



class CNN_model():
    def __init__(self, n_in, n_out, data, batch_size, rng, learning_rate, activation=T.tanh, L1_reg=0., L2_reg=0.0001):
        index = T.lscalar()  # index to a [mini]batch
        x = T.matrix('x')   
        y = T.ivector('y')  
        X = T.matrix('X')  
        Y = T.ivector('Y')                             
        
        if (rng is None):
            rng = np.random.RandomState(1234)        
        self.batch_size = batch_size
        self.n_train_batches = data.train.x.get_value(borrow=True).shape[0] / batch_size
        self.n_valid_batches = data.valid.x.get_value(borrow=True).shape[0] / batch_size
        self.n_test_batches  = data.test.x.get_value (borrow=True).shape[0] / batch_size        
        
        self.x = x        
        self.y = y
        self.data = data 
        
        
        # Define the layers    
        self.convLayer = hlv_layers.ConvLayer(      rng =  rng, 
                                                    input =  x.reshape((-1, 1, 64, 64)), 
                                                    filter_shape = (16, 1, 25, 25), 
                                                    image_shape = (batch_size, 1, 64, 64), 
                                                    activation=T.tanh)
        self.logRegressionLayer = hlv_layers.LogisticRegression(
                                    input=self.convLayer.output.flatten(2),
                                    n_in=40*40*16,
                                    n_out=n_out)
        
        
        # Define regularization
        self.L1 = abs(self.convLayer.W).sum() \
                + abs(self.logRegressionLayer.W).sum()
        self.L2_sqr = (self.convLayer.W ** 2).sum() \
                    + (self.logRegressionLayer.W ** 2).sum()

        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.errors = self.logRegressionLayer.errors

        # define the cost
        self.cost = self.negative_log_likelihood(y)
            #\
            # + L1_reg * self.L1 \
            # + L2_reg * self.L2_sqr
    
        # define parameters
        self.params = self.convLayer.params + self.logRegressionLayer.params
 
        self.grads = [T.grad(cost=self.cost, wrt=param) for param in self.params]
        
        self.updates = [(param, param - learning_rate * grad) for
                    param, grad in zip(self.params, self.grads)]
    
        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`

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
        self.train_model_minibatch = theano.function(inputs=[index],
                                            outputs=self.cost,
                                            updates=self.updates,
                                            givens={
                                                x: self.data.train.x[index * self.batch_size:(index + 1) * self.batch_size],
                                                y: self.data.train.y[index * self.batch_size:(index + 1) * self.batch_size]})                  
reload(hlv_layers)
reload(hlv_models)
lr = 0.1
lr_exp = 0.5
batch_size = 300
n_hidden = 100
print "compiling model"
print "model compiled"


M = CNN_model(64*64, 2, data_faces, batch_size, None, lr)
param_values = hlv_aux.get_params(M)
while lr > 0.00001:
    print "============ LR %f" % (lr)
    M = CNN_model(64*64, 2, data_faces, batch_size, None, lr)
    #M = hlv_models.MLP_model(64*64, 2, data_faces, batch_size, None, n_hidden, lr)
    hlv_aux.set_params(M, param_values)
    param_values = hlv_train.train_minibatches(M, max_epochs=100)
    lr = lr * lr_exp

