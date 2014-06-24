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

showBunch(param_values[2], 1, 64)   


print "Loading data ..."
t0 = time.time()
data_faces = faces_data()
t1 = time.time()
print "Data loaded in %0.2f seconds" % ( t1-t0)


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
lr = 0.1
lr_exp = 0.5
batch_size = 300
#prob_drop = 0.333
n_hidden = [30,20,10]
M = hlv_models.MLPM_model(64*64, 2, data_faces, batch_size, None, n_hidden, 1,L2_reg=0.1)

#M = hlv_models.MLP_model(64*64, 2, data_faces, batch_size, None, n_hidden, 1)
param_values = hlv_aux.get_params(M)
while lr > 0.0001:
    print "============ LR %f" % (lr)
    M = hlv_models.MLPM_model(64*64, 2, data_faces, batch_size, None, n_hidden, lr, L2_reg=0.1)
    #M = hlv_models.MLP_model(64*64, 2, data_faces, batch_size, None, n_hidden, lr)
    hlv_aux.set_params(M, param_values)
    param_values = hlv_train.train_minibatches(M, min_epochs=250, max_epochs=1000)
    lr = lr * lr_exp
    
    
    