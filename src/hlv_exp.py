# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 18:05:40 2014

@author: tc
"""

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
from theano import pp

from hooloovoo_aux import load_data
import hooloovoo_init as hlv_init
import hooloovoo_layers as hlv_layers
import hooloovoo_tests
import hooloovoo_train as hlv_train


rectifier = lambda x: T.maximum(0, x)

def show(x, r=1, size=64):
    q.show((x.reshape(size,size)-np.min(x)) / (np.max(x) - np.min(x)), r=r)

# Get the data
def faces_data():
    faces, labels = q.load_from_pkl("/home/tc/faces.bzpkl")
    labels = np.asarray(labels, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.float64)
    
    data = q.newObject()
    data.train = q.newObject()
    data.valid = q.newObject()
    data.test = q.newObject()
    
    data.train.x = np.vstack((faces[labels==1,:][:1000], faces[labels==-1,:][:1000]))
    data.valid.x = np.vstack((faces[labels==1,:][1000:1500], faces[labels==-1,:][1000:1500]))
    data.test.x = np.vstack((faces[labels==1,:][1500:2000], faces[labels==-1,:][1500:2000]))
    data.train.y = np.hstack((np.ones((1000, ), dtype=np.int32),
                              np.zeros((1000, ),dtype=np.int32)))
    data.valid.y = np.hstack((np.ones((500, ),dtype=np.int32),
                              np.zeros((500, ),dtype=np.int32)))
    data.test.y  = np.hstack((np.ones((500, ),dtype=np.int32),
                              np.zeros((500, ),dtype=np.int32)))
    
    data.train.x = theano.shared(data.train.x, borrow=True)
    data.train.y = theano.shared(data.train.y, borrow=True)
    data.valid.x = theano.shared(data.valid.x, borrow=True)
    data.valid.y = theano.shared(data.valid.y, borrow=True)
    data.test.x  = theano.shared(data.test.x , borrow=True)
    data.test.y  = theano.shared(data.test.y , borrow=True)
    #np.random.seed(10)
    #np.random.shuffle(data.train.x)
    #np.random.seed(10)
    #np.random.shuffle(data.train.y)
    
    
    def show(img):
        q.show(img.reshape((64,64)))
        cv2.destroyAllWindows()
        for k in range(10):
            cv2.waitKey(10)
    data.show = show
    return data
        




class LogisticRegression_model():
    def __init__(self, n_in, n_out, data, learning_rate = 0.13, batch_size=1):        
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
        self.train_model_minibatch = theano.function(inputs=[index],
                                            outputs=self.cost,
                                            updates=self.updates,
                                            givens={
                                                x: self.data.train.x[index * self.batch_size:(index + 1) * self.batch_size],
                                                y: self.data.train.y[index * self.batch_size:(index + 1) * self.batch_size]})                  
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
    def train(self,x,y):        
        return self.train_model(x, y)
    
    def train_epoch(self):
        return self.train(self.data.train.x, self.data.train.y)
        
    def train_epochs(self,n_epochs=1000):        
        best_loss = np.inf
        tloss = np.inf
        for i in range(n_epochs):
            loss = self.train_epoch()
            print "epoch",i,"training error:"  ,  loss
            vloss = self.validate_model()
            print "                validation error:", vloss
            if vloss < best_loss:
                best_loss = vloss
                tloss = self.test_model()
                best_params = []
                for param in self.params:
                    best_params.append(param.get_value())            
                print "                               test error:", tloss
        print "Final Test Error", tloss
        
    def train_minibatches(self,     max_epochs = 1000, 
                                    patience_increase=2, 
                                    improvement_threshold=0.995,
                                    validation_frequency=None):
        print '... training the model'
        
        if validation_frequency==None:
            validation_frequency=self.n_train_batches
        # initialization parameters
        best_params = None
        best_validation_loss = np.inf
        test_score = 0.
        start_time = time.clock()
        epoch = 0
        done_looping = False
        patience = self.n_train_batches * 10
        
        while (not done_looping):
            
            for batch_idx  in xrange(self.n_train_batches):
                # Train
                batch_avg_cost = self.train_model_minibatch(batch_idx)
                print batch_avg_cost
                iteration = epoch * self.n_train_batches + batch_idx
                #print "batch", batch_idx
                # validation if right time
                if (iteration + 1) % validation_frequency == 0:
                    validation_loss = self.validate_model()
                    print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                        (epoch, batch_idx + 1, self.n_train_batches,
                         validation_loss * 100.))
                    if validation_loss < best_validation_loss * improvement_threshold:                                                
                        patience = max(patience, iteration * patience_increase)
                        best_validation_loss = validation_loss
                        test_score = self.test_model()
                        print(('     epoch %i, minibatch %i/%i, test error of best'
                            ' model %f %%') % (epoch, batch_idx + 1, 
                            self.n_train_batches, test_score * 100.))
            # check if done looping
            epoch = epoch + 1
            if epoch == max_epochs or patience <= iteration:
                done_looping = True
        end_time = time.clock()
        print(('Optimization complete with best validation score of %f %%,'
               'with test performance %f %%') %
               (best_validation_loss * 100., test_score * 100.))
        print 'The code run for %d epochs, with %f epochs/sec' % (
            epoch, 1. * epoch / (end_time - start_time))
        print ('The code ran for %.1fs' % ((end_time - start_time)))                               
    



data_faces = faces_data()
mnist = load_data('mnist.pkl.gz')

data_digits = q.newObject()
data_digits.train = q.newObject()
data_digits.valid = q.newObject()
data_digits.test = q.newObject()

data_digits.train.x = mnist[0][0]
data_digits.valid.x = mnist[1][0]
data_digits.test.x  = mnist[2][0]

data_digits.train.y = mnist[0][1]
data_digits.valid.y = mnist[1][1]
data_digits.test.y  = mnist[2][1]


print "Ready, Cap\'n!"


M = LogisticRegression_model(28*28, 10, data_digits, batch_size=600,learning_rate = 0.13)
M.train_minibatches()
M.validate_model()
M.test_model()

M2 = LogisticRegression_model(64*64, 2, data_faces, batch_size=100,learning_rate = 0.13)
M2.train_epochs(10)
M2.validate_model()
M2.test_model()



"""




M = LogisticRegression_model(64*64, 2, data_faces, batch_size=500,learning_rate = 0.01)
hlv_train.train_minibatches(M)
hlv_train.train_epochs(M,100)
M.validate_model()
M.test_model()


M2 = LogisticRegression_model(n_in=28*28, n_out=2, data=data_digits, batch_size=20, learning_rate=0.01)
hlv_train.train_minibatches(M2)


hlv_train.train_minibatches(M2)


M = LogisticRegression_model(28*28, 2, data, batch_size=50,learning_rate = 0.01)
hlv_train.train_minibatch(M,3)
hlv_train.train_epochs(M,100)
M.validate_model()
M.test_model()




M = LogisticRegression_model(64*64, 2, data1, batch_size=500,learning_rate = 0.01)
hlv_train.train_minibatches(M)
hlv_train.train_epochs(M,100)
M.validate_model()
M.test_model()

M = LogisticRegression_model(28*28, 10, data, batch_size=20,learning_rate = 0.01)
hlv_train.train_minibatches(M)
hlv_train.train_epochs(M,100)
M.validate_model()
M.test_model()
"""