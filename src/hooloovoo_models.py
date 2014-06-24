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

from hooloovoo_aux import mnist_data
import hooloovoo_aux as hlv_aux
import hooloovoo_init as hlv_init
import hooloovoo_layers as hlv_layers
import hooloovoo_tests
import hooloovoo_train as hlv_train


class LogisticRegression_model():
    def __init__(self, n_in, n_out, data, learning_rate = 0.001, batch_size=1):
        index = T.lscalar()  # index to a [mini]batch        
        X = T.matrix()  
        Y = T.ivector()  
        x = T.matrix('x')  
        y = T.ivector('y')  
        
        self.batch_size = batch_size
        self.n_train_batches = data.train.x.get_value(borrow=True).shape[0] / batch_size
        self.n_valid_batches = data.valid.x.get_value(borrow=True).shape[0] / batch_size
        self.n_test_batches  = data.test.x.get_value (borrow=True).shape[0] / batch_size        
        
        self.x = x        
        self.y = y
        self.data = data
        self.classifier = hlv_layers.LogisticRegression(input=self.x, 
                                     n_in=n_in, n_out=n_out)
        self.cost = self.classifier.negative_log_likelihood(y)
        
        
        self.params = [self.classifier.W, self.classifier.b]
        
        self.grads = [T.grad(cost=self.cost, wrt=param) for param in self.params]
        
        self.updates = [(param, param - learning_rate * grad) for
                    param, grad in zip(self.params, self.grads)]
                   
                  
        # Model operations (training/testing)
        self.train_model = theano.function(inputs=[X, Y],
                                            outputs=self.cost,
                                            updates=self.updates,
                                            givens={
                                                x: X,
                                                y: Y})
        self.test_model = theano.function(  inputs=[],
                                            outputs=self.classifier.errors(y),
                                            givens={
                                                x: self.data.test.x,
                                                y: self.data.test.y})
                                                
        self.validate_model = theano.function(inputs=[],
                                            outputs=self.classifier.errors(y),
                                            givens={
                                                x: self.data.valid.x,
                                                y: self.data.valid.y})
        self.train_model_minibatch = theano.function(inputs=[index],
                                            outputs=self.cost,
                                            updates=self.updates,
                                            givens={
                                                x: self.data.train.x[index * self.batch_size:(index + 1) * self.batch_size],
                                                y: self.data.train.y[index * self.batch_size:(index + 1) * self.batch_size]})                  
        
                                                
"""
Class Model 
    Should expose    
        train_model(X, Y)
        test_model()
        validate_model()
        train_model_minibatch(minibatch_index)
        cost
        params
    Should get (as parameters)
        n_in
        n_out
        data
        batch_size
        rng
        Learning parameters:
            learning rate
            L1_reg
            L2_reg
            activation function
    Should keep for itself:
        internal model structure (like layers)
                
    
    
"""


class MLP_model_dropout2():
    def __init__(self, n_in, n_out, data, batch_size, rng, n_hidden, learning_rate, activation=T.tanh, L1_reg=0., L2_reg=0.0001, prob_drop=0.2):
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
        # Define the drop out layers
        self.dropInLayer = hlv_layers.DropOutLayer(x, n_in)
        self.hiddenLayer_do = hlv_layers.HiddenLayer(rng=rng, input=self.dropInLayer.output,
                                                  n_in=n_in, n_out=n_hidden,
                                                  activation=activation)
        self.dropOutLayer = hlv_layers.DropOutLayer(self.hiddenLayer_do.output, n_hidden)
        self.logRegressionLayer_do = hlv_layers.LogisticRegression(
                                    input=self.dropOutLayer.output,
                                    n_in=n_hidden,
                                    n_out=n_out)


        # define the vanilla layers
        self.hiddenLayer = hlv_layers.HiddenLayer(rng=rng, input=x,
                                                  n_in=n_in, n_out=n_hidden,
                                                  activation=activation)
        self.hiddenLayer.W = self.hiddenLayer_do.W
        self.hiddenLayer.b = self.hiddenLayer_do.b
        lin_output = T.dot(x, self.hiddenLayer.W) + self.hiddenLayer.b
        self.hiddenLayer.output = (lin_output if activation is None
                       else activation(lin_output))



        self.logRegressionLayer = hlv_layers.LogisticRegression(
                                    input=self.hiddenLayer.output,
                                    n_in=n_hidden,
                                    n_out=n_out)
        # Make the vanilla logistic regression layer have the same parameters
        #    as the dropout layer 
        
        self.cost = 0
        self.logRegressionLayer.W = self.logRegressionLayer_do.W
        self.logRegressionLayer.b = self.logRegressionLayer_do.b
        self.logRegressionLayer.p_y_given_x = T.nnet.softmax(T.dot(self.hiddenLayer.output, self.logRegressionLayer.W) + self.logRegressionLayer.b)
        self.logRegressionLayer.y_pred = T.argmax(self.logRegressionLayer.p_y_given_x, axis=1)
        
        
        # Define regularization
        self.L1 = abs(self.hiddenLayer.W).sum() \
                + abs(self.logRegressionLayer_do.W).sum()
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
                    + (self.logRegressionLayer_do.W ** 2).sum()

        self.negative_log_likelihood = self.logRegressionLayer_do.negative_log_likelihood
        self.errors = self.logRegressionLayer.errors

        # define the cost
        self.cost = self.negative_log_likelihood(y) \
             + L1_reg * self.L1 \
             + L2_reg * self.L2_sqr
    
        # define parameters
        self.params = self.hiddenLayer_do.params + self.logRegressionLayer_do.params
 
        self.grads = [T.grad(cost=self.cost, wrt=param) for param in self.params]
        
        self.updates = [(param, param - learning_rate * grad) for
                    param, grad in zip(self.params, self.grads)]
    
        #  Just dropout stuff
        ##########################
        self.prob_drop = prob_drop
        self.srng = theano.tensor.shared_randomstreams.RandomStreams(0)
        def random_drop_mask(size):
            # p=1-p because 1's indicate keep and p is prob of dropping
            mask = self.srng.binomial(n=1, p=1-prob_drop, size=size)
            # The cast is important because
            # int * float32 = float64 which pulls things off the gpu
            output = T.cast(mask, theano.config.floatX)
            return output
        self.dropInLayer.drop_mask.set_value(random_drop_mask(self.dropInLayer.drop_mask.shape).eval())
        self.dropOutLayer.drop_mask.set_value(random_drop_mask(self.dropOutLayer.drop_mask.shape).eval())
        self.updates.append((self.dropInLayer.drop_mask, random_drop_mask(self.dropInLayer.drop_mask.shape)))    
        self.updates.append((self.dropOutLayer.drop_mask, random_drop_mask(self.dropOutLayer.drop_mask.shape)))    
    
    
    
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
        

      

class MLP_model_dropout():
    def __init__(self, n_in, n_out, data, batch_size, rng, n_hidden, learning_rate, activation=T.tanh, L1_reg=0., L2_reg=0.0001, prob_drop=0.2):
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
        self.hiddenLayer = hlv_layers.HiddenLayer(rng=rng, input=x,
                                                  n_in=n_in, n_out=n_hidden,
                                                  activation=activation)
        self.dropOutLayer = hlv_layers.DropOutLayer(self.hiddenLayer.output, n_hidden)
        self.logRegressionLayer_do = hlv_layers.LogisticRegression(
                                    input=self.dropOutLayer.output,
                                    n_in=n_hidden,
                                    n_out=n_out)
        self.logRegressionLayer = hlv_layers.LogisticRegression(
                                    input=self.hiddenLayer.output,
                                    n_in=n_hidden,
                                    n_out=n_out)
        # Make the vanilla logistic regression layer have the same parameters
        #    as the dropout layer 
        self.logRegressionLayer.W = self.logRegressionLayer_do.W
        self.logRegressionLayer.b = self.logRegressionLayer_do.b
        self.logRegressionLayer.p_y_given_x = T.nnet.softmax(T.dot(self.hiddenLayer.output, self.logRegressionLayer.W) + self.logRegressionLayer.b)
        self.logRegressionLayer.y_pred = T.argmax(self.logRegressionLayer.p_y_given_x, axis=1)
        
        
        # Define regularization
        self.L1 = abs(self.hiddenLayer.W).sum() \
                + abs(self.logRegressionLayer_do.W).sum()
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
                    + (self.logRegressionLayer_do.W ** 2).sum()

        self.negative_log_likelihood = self.logRegressionLayer_do.negative_log_likelihood
        self.errors = self.logRegressionLayer.errors

        # define the cost
        self.cost = self.negative_log_likelihood(y) \
             + L1_reg * self.L1 \
             + L2_reg * self.L2_sqr
    
        # define parameters
        self.params = self.hiddenLayer.params + self.logRegressionLayer_do.params
 
        self.grads = [T.grad(cost=self.cost, wrt=param) for param in self.params]
        
        self.updates = [(param, param - learning_rate * grad) for
                    param, grad in zip(self.params, self.grads)]
    
        #  Just dropout stuff
        ##########################
        self.prob_drop = prob_drop
        self.srng = theano.tensor.shared_randomstreams.RandomStreams(0)
        def random_drop_mask():
            # p=1-p because 1's indicate keep and p is prob of dropping
            mask = self.srng.binomial(n=1, p=1-prob_drop, size=self.dropOutLayer.drop_mask.shape)
            # The cast is important because
            # int * float32 = float64 which pulls things off the gpu
            output = T.cast(mask, theano.config.floatX)
            return output
        self.dropOutLayer.drop_mask.set_value(random_drop_mask().eval())
        self.updates.append((self.dropOutLayer.drop_mask, random_drop_mask()))    
    
    
    
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
        

      

class MLP_model():
    def __init__(self, n_in, n_out, data, batch_size, rng, n_hidden, learning_rate, activation=T.tanh, L1_reg=0., L2_reg=0.0001):
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
        self.hiddenLayer = hlv_layers.HiddenLayer(rng=rng, input=x,
                                                  n_in=n_in, n_out=n_hidden,
                                                  activation=activation)
        self.logRegressionLayer = hlv_layers.LogisticRegression(
                                    input=self.hiddenLayer.output,
                                    n_in=n_hidden,
                                    n_out=n_out)
        
        
        # Define regularization
        self.L1 = abs(self.hiddenLayer.W).sum() \
                + abs(self.logRegressionLayer.W).sum()
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
                    + (self.logRegressionLayer.W ** 2).sum()

        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.errors = self.logRegressionLayer.errors

        # define the cost
        self.cost = self.negative_log_likelihood(y) \
             + L1_reg * self.L1 \
             + L2_reg * self.L2_sqr
    
        # define parameters
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
 
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
        



class MLPM_model():
    def __init__(self, n_in, n_out, data, batch_size, rng, n_hidden, learning_rate, activation=T.tanh, L1_reg=0., L2_reg=0.0001):
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
        layer_input = x
        layer_n_in = n_in
        self.hiddenLayers = []
        for i in range(len(n_hidden)-1):
            layer_n_out = n_hidden[i+1]
            hiddenLayer = hlv_layers.HiddenLayer(rng=rng, input=layer_input,
                                                      n_in=layer_n_in, n_out=layer_n_out,
                                                      activation=activation)
            self.hiddenLayers.append(hiddenLayer)
            layer_input = hiddenLayer.output
            layer_n_in = layer_n_out
            
            
        self.logRegressionLayer = hlv_layers.LogisticRegression(
                                    input=layer_input,
                                    n_in=layer_n_in,
                                    n_out=n_out)
        
        
        # Define regularization
        self.L1 = 0
        self.L2_sqr = 0
        # define parameters
        self.params =  self.logRegressionLayer.params

        for HL in self.hiddenLayers:
            self.L1 = self.L1 + abs(HL.W).sum()
            self.L2 = self.L2_sqr + abs(HL.W**2).sum()

            self.params = self.params + HL.params 



        self.L1 = self.L1 + abs(self.logRegressionLayer.W).sum()
        self.L2_sqr = self.L2_sqr + (self.logRegressionLayer.W ** 2).sum()

        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.errors = self.logRegressionLayer.errors

        # define the cost
        self.cost = self.negative_log_likelihood(y) \
             + L1_reg * self.L1 \
             + L2_reg * self.L2_sqr
    
 
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
        
