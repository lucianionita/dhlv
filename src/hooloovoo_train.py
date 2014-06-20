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
from hooloovoo_aux import load_data
import hooloovoo_init as hlv_init
import hooloovoo_layers as hlv_layers
import hooloovoo_tests

def train(self,x,y):        
    return self.train_model(x.eval(),y.eval())

def train_epoch(self):
    train(self, self.data.train.x, self.data.train.y)
    
def train_minibatch(self, idx):        
    X = self.data.train.x[idx * self.batch_size:(idx + 1) * self.batch_size]
    Y = self.data.train.y[idx * self.batch_size:(idx + 1) * self.batch_size]
    print X.eval(), Y.eval()
    return train(self, X, Y)        
    
def train_epochs(self,n_epochs=1000):        
    best_loss = np.inf
    tloss = np.inf
    for i in range(n_epochs):
        loss = train_epoch(self)
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
            batch_avg_cost = train_minibatch(self,batch_idx)
            iteration = epoch * self.n_train_batches + batch_idx
            
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