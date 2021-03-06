import os
import cPickle as pickle
import cPickle
#import cv2
import quick as q
import theano
import sys
import time
import gzip
import numpy as np
import theano

#theano.config.floatX  ='float64'

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


import voo

rectifier = lambda x: T.maximum(0, x)


"""
if 'data_faces' in dir(): 
    print "... Data seems to be already loaded"
else:
    data_faces = faces_data()
"""    
if 'data_digits' in dir(): 
    print "... Data seems to be already loaded"
else:
    data_digits = mnist_data()

# NOTE, the internal input shape is without regard to BATCHES
# input shape must be a tuple
# output shape is also a tuple
import warnings


        


class Generic_model():
    def __init__(self, n_in, n_out, data, layerSpecs, batch_size, rng, learning_rate, activation=T.tanh, L1_reg=0., L2_reg=0.0001):

        # Define classifier independent stuff
        # preliminaries
        #######################################################################

        index = T.lscalar()  # index to a [mini]batch
        x = T.matrix('x')
        y = T.ivector('y')  
        X = T.matrix('X')  
        Y = T.ivector('Y')       
        self.lr = theano.shared(learning_rate)
        
        if (rng is None):
            rng = np.random.RandomState(23455)  
            
        self.batch_size = batch_size
        self.n_train_batches = data.train.x.get_value(borrow=True).shape[0] / batch_size
        self.n_valid_batches = data.valid.x.get_value(borrow=True).shape[0] / batch_size
        self.n_test_batches  = data.test.x.get_value (borrow=True).shape[0] / batch_size        
        
        self.data = data 
        
        
        # Define the model structure    
        #######################################################################
        
        layerClasses = {
            'conv': voo.layers.ConvLayer,
            'hidden': voo.layers.FullyConnectedLayer,
            'hidden_decomp': voo.layers.decomp.FullyConnectedDecompLayer,
            'logistic': voo.layers.LogisticRegressionLayer,
            'pooling': voo.layers.PoolingLayer,
            'dropout': voo.layers.DropOutLayer
            }
        
        Layers = []

        input_layer = voo.layers.InputLayer(x, n_in, batch_size)
        Layers.append(input_layer)
        prev_layer = input_layer
        for layer_idx in range(len(layerSpecs)):
            # First we get the layer class and specs
            layerType, layerConfig = layerSpecs[layer_idx]
            layerClass = layerClasses[layerType]           
            new_layer = voo.layers.GenLayer(layerClass, prev_layer, batch_size, rng, layerConfig)
            
            Layers.append(new_layer)
            prev_layer = new_layer
        
        self.Layers=Layers

        #
        # Define all the model specifics that relate to the parameters:
        # vector of parameters, gradients, errors, updates, accuracy cost
        # regularization cost, overall cost
        ######################################################################


        self.params = []
        for layer in Layers:
            self.params.extend(layer.params)
        

        self.acc_cost = self.Layers[-1].negative_log_likelihood(y)
        self.L1 = np.sum([abs(param).sum() for param in self.params])
        self.L2 = np.sum([(param**2).sum() for param in self.params])
        self.reg_cost = L1_reg * self.L1 + L2_reg * self.L2
        self.cost = self.acc_cost + self.reg_cost



        self.grads = [T.grad(cost=self.cost, wrt=param) for param in self.params]    
        
        self.updates = [(param, param - self.lr * grad) for
                    param, grad in zip(self.params, self.grads)]



        

        # Define the model training and testing functions for both
        # minibatches and arbitrary data
    
        self.train_model = theano.function(inputs=[X, Y],
                                            outputs=self.errors(y),
                                            updates=self.updates,
                                            givens={
                                                x: X,
                                                y: Y})
        self.test_model2 = theano.function(  inputs=[],
                                            outputs=self.errors(y),
                                            givens={
                                                x: self.data.test.x,
                                                y: self.data.test.y})
                                                
        self.validate_model2 = theano.function(inputs=[],
                                            outputs=self.errors(y),
                                            givens={
                                                x: self.data.valid.x,
                                                y: self.data.valid.y}) 
                                                        
        self.minibatch = q.newObject()        
        self.minibatch.test = theano.function(  inputs=[index],
                                            outputs=self.errors(y),
                                            givens={
                                                x: self.data.test.x[index * self.batch_size:(index + 1) * self.batch_size],
                                                y: self.data.test.y[index * self.batch_size:(index + 1) * self.batch_size]})
                                                
        self.minibatch.validate = theano.function(inputs=[index],
                                            outputs=self.errors(y),
                                            givens={
                                                x: self.data.valid.x[index * self.batch_size:(index + 1) * self.batch_size],
                                                y: self.data.valid.y[index * self.batch_size:(index + 1) * self.batch_size]}) 
        self.minibatch.train = theano.function(inputs=[index],
                                            outputs=self.cost,
                                            updates=self.updates,
                                            givens={
                                                x: self.data.train.x[index * self.batch_size:(index + 1) * self.batch_size],
                                                y: self.data.train.y[index * self.batch_size:(index + 1) * self.batch_size]})                  
    def validate_model(self):
        validation_losses = [self.minibatch.validate(i) for i
                                     in xrange(self.n_valid_batches)]
        return np.mean(validation_losses)
    def test_model(self):
        test_losses = [self.minibatch.test(i) for i
                                     in xrange(self.n_test_batches)]
        return np.mean(test_losses) 
    def errors(self, y):
        return self.Layers[-1].errors(y)
    def randomize(self):
        for layer in self.Layers:
            if 'reset-randomize' in layer.options:
                layer.randomize()
    def reset(self):
        for layer in self.Layers:
            if 'reset-randomize' in layer.options:
                layer.reset()


def Train_minibatches(self,     min_epochs = 100,
                                max_epochs = 1000, 
                                patience_increase=2, 
                                improvement_threshold=0.995,
                                validation_frequency=None):
    print '... training the model with minibatches'
    
    if validation_frequency==None:
        validation_frequency=self.n_train_batches
    
    # initialization parameters
    best_params = hlv_aux.get_params(self)
    best_validation_loss = np.inf
    test_score = 0.
    start_time = time.time()
    epoch = 0
    done_looping = False
    patience = self.n_train_batches * min_epochs
    
    print "Before we start:"
    best_validation_loss = self.validate_model()    
    test_score = self.test_model()    
    print "     Validation score", best_validation_loss
    print "     Test       score", test_score
    
    while (not done_looping):
        t0 = time.time()
        for batch_idx  in xrange(self.n_train_batches):
            # Train            
            batch_avg_cost = self.minibatch.train(batch_idx)
            iteration = epoch * self.n_train_batches + batch_idx
            t1 = time.time()
            sys.stdout.write("Training batch %i/%i, Time(elapsed/estimated) %.0fs/%.0fs\r" %(batch_idx+1, self.n_train_batches, t1-t0, (t1-t0)/(batch_idx+1)*self.n_train_batches))
            sys.stdout.flush()
            # validation if right time
            if (iteration + 1) % validation_frequency == 0:
                # get validation loss
                validation_loss = self.validate_model()
                print('\nepoch %i, mb %i/%i, tcost %.5f, verror %.3f%%' % \
                    (epoch, batch_idx + 1, self.n_train_batches,
                     batch_avg_cost, validation_loss * 100.))
                # Check if validation error is worth looking at
                if validation_loss < best_validation_loss * improvement_threshold:                                                
                    patience = max(patience, iteration * patience_increase)
                    best_validation_loss = validation_loss
                    test_score = self.test_model()
                    print(('     epoch %i, minibatch %i/%i, test error of best'
                        ' model %0.3f%%') % (epoch, batch_idx + 1, 
                        self.n_train_batches, test_score * 100.))
                    best_params = hlv_aux.get_params(self)
        # check if done looping
        epoch = epoch + 1
        if epoch == max_epochs or patience <= iteration:
            done_looping = True
    end_time = time.time()
    print "Optimization Complete!"
    print "Best validation error %.3f%%. Test Error: %.3f%%" % (best_validation_loss * 100., test_score * 100.)
    print 'The training ran for %d epochs, with %f epochs/sec' % ( epoch, 1. * epoch / (end_time - start_time))
    return best_params

reload(hlv_layers)
reload(hlv_models)
reload(hlv_train)
lr = 0.1
lr_exp = 0.1

batch_size = 500

specs = [( 'conv'  ,    {	'n_filters':32,     
				'filter':(5,5),
			}), 
         ( 'pooling',   {	'poolsize':(2,2)	
			}),
         ( 'conv'  ,    {	'n_filters':64,     
				'filter':(3,3)
			}), 
         ( 'pooling',   {	'poolsize':(2,2)	
			}),
         ( 'conv'  ,    {	'n_filters':128,     
				'filter':(2,2)
			}), 
         ( 'pooling',   {	'poolsize':(2,2)
			}),
         ( 'hidden'  ,  {	'n_out':100
			}), 
#         ( 'dropout',   {	'prob':0.5
#			}), 
         ( 'logistic',  {	'n_out':10
			})
]
M = Generic_model(n_in = (28,28), n_out = 10, data= data_digits, layerSpecs = specs, 
                   batch_size=batch_size, rng=None, learning_rate=lr, activation=rectifier, 
                   L1_reg=0., L2_reg=0.0001)

#M.minibatch.validate(0)
#print "Model Magnitude:", hlv_aux.Magnitude(M)
#Train_minibatches(M, min_epochs=5, max_epochs=10)

param_values = hlv_aux.get_params(M)
#param_values = hlv_train.train_minibatches(M, min_epochs=20, max_epochs=200)

while lr > 0.0001:
    print "============ LR %f" % (lr)
    M = Generic_model(n_in = (28,28), n_out = 10, data= data_digits, layerSpecs = specs, 
                   batch_size=batch_size, rng=None, learning_rate=lr, activation=rectifier, 
                   L1_reg=0., L2_reg=0.0001)
    #M = hlv_models.MLP_model(64*64, 2, data_faces, batch_size, None, n_hidden, lr)
    hlv_aux.set_params(M, param_values)
    #param_values = hlv_train.train_minibatches(M, min_epochs=20, max_epochs=200)
    param_values = Train_minibatches(M, min_epochs=10, max_epochs=100)
    lr = lr * lr_exp

