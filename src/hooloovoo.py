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
    def __init__(self, n_in, n_out, data, learning_rate = 0.001, batch_size=1):
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
                                            outputs=self.classifier.errors(Y),
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
                                                
"""
Class Model 
    Should expose    
        train_model(X, Y)
        test_model()
        validate_model()
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


class MLP_model():
    def __init__(self, n_in, n_out, data, batch_size, rng, n_hidden, learning_rate, activation=T.tanh, L1_reg=0., L2_reg=0.00001):
        
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
        
        updates = [(param, param - learning_rate * grad) for
                    param, grad in zip(self.params, self.grads)]
    
        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`

        self.train_model = theano.function(inputs=[X, Y],
                                            outputs=self.errors(y),
                                            updates=updates,
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