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
import warnings


        






reload(hlv_layers)
reload(hlv_models)
reload(hlv_train)
lr = 0.01
lr_exp = 0.5
L2_reg = 0.0
batch_size = 50

"""
         ( 'conv'  ,    {	'n_filters':16,     
				'filter':(7,7)
			}), 
         ( 'pooling',   {	'poolsize':(3,3)	
			}),
         ( 'conv'  ,    {	'n_filters':32,     
				'filter':(4,4)
			}), 
         ( 'pooling',   {	'poolsize':(2,2)	
			}),
         ( 'conv'  ,    {	'n_filters':64,     
				'filter':(2,2)
			}), 
         ( 'pooling',   {	'poolsize':(2,2)
			}),
         ( 'hidden'  ,  {	'n_out':100
			}), 
"""
#         ( 'dropout',   {	'prob':0.5
#			}), 
specs = [
	 (  'conv'   ,  {	'n_filters':	128,
				'filter':	(7,7),
			}), 
         ( 'pooling',   {	'poolsize':	(2,2)
			}),
	 (  'conv'   ,  {	'n_filters':	256,
				'filter':	(5,5),
			}), 
         ( 'pooling',   {	'poolsize':	(2,2)
			}),
	 (  'conv'   ,  {	'n_filters':	512,
				'filter':	(4,4),
			}), 
         ( 'pooling',   {	'poolsize':	(2,2)
			}),
         ( 'hidden'  ,  {	'n_out':1000
			}), 
         ( 'hidden'  ,  {	'n_out':250
			}), 
         ( 'hidden'  ,  {	'n_out':50
			}), 
         ( 'logistic',  {	'n_out':2
			})
]
specs = [
	 (  'conv'   ,  {	'n_filters':	32,
				'filter':	(9,9),
			}), 
         ( 'pooling',   {	'poolsize':	(2,2)
			}),
	 (  'conv'   ,  {	'n_filters':	64,
				'filter':	(5,5),
			}), 
         ( 'pooling',   {	'poolsize':	(2,2)
			}),
	 (  'conv'   ,  {	'n_filters':	128,
				'filter':	(3,3),
			}), 
         ( 'pooling',   {	'poolsize':	(2,2)
			}),
         ( 'hidden'  ,  {	'n_out':500
			}), 
         ( 'hidden'  ,  {	'n_out':50
			}), 
         ( 'logistic',  {	'n_out':2
			})
]
specs =  [
         ( 'conv'  ,    {	'n_filters':12,
				'filter':(7,7)
			}), 
         ( 'pooling',   {	'poolsize':(2,2)	
			}),
         ( 'conv'  ,    {	'n_filters':24,  
				'filter':(5,5)
			}), 
         ( 'pooling',   {	'poolsize':(2,2)	
			}),
         ( 'conv'  ,    {	'n_filters':48,
				'filter':(3,3)
			}), 
         ( 'pooling',   {	'poolsize':(2,2)
			}),
         ( 'hidden'  ,  {	'n_out':10
			}),
         ( 'logistic',  {	'n_out':2
			})
]
M = Generic_model(n_in = (64,64), n_out = 2, data= data_faces, layerSpecs = specs, 
                   batch_size=batch_size, rng=None, learning_rate=lr, activation=T.tanh, 
                   L1_reg=0., L2_reg=L2_reg)

print "Model Magnitude:", np.log(np.sum([np.prod(p.eval().shape) for p in M.params]))/np.log(2)

M.lr.set_value(np.float32(lr))
param_values = Train_minibatches(M, min_epochs=200, max_epochs=1500, validation_frequency=41)
"""
param_values = hlv_aux.get_params(M)

while lr > 0.00005:
    print "============ LR %f" % (lr)
    #hlv_aux.set_params(M, param_values)
    #param_values = hlv_train.train_minibatches(M, min_epochs=20, max_epochs=200)
    param_values = Train_minibatches(M, min_epochs=50, max_epochs=1000, validation_frequency=39)
    lr = lr * lr_exp
"""
