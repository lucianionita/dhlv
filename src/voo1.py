import os
import cPickle as pickle
import quick as q
import theano
import sys
import time
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import voo

"""
if 'data_faces' in dir(): 
    print "... Data seems to be already loaded"
else:
    data = voo.datasets.get_faces()
"""    
if 'data_digits' in dir(): 
    print "... Data seems to be already loaded"
else:
    data = voo.datasets.get_mnist()

import warnings


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
         ( 'conv'  ,    {	'n_filters':4,
				'filter':(7,7)
			}), 
         ( 'pooling',   {	'poolsize':(2,2)	
			}),
         ( 'conv'  ,    {	'n_filters':4,  
				'filter':(5,5)
			}), 
         ( 'pooling',   {	'poolsize':(2,2)	
			}),
         ( 'conv'  ,    {	'n_filters':4,
				'filter':(3,3)
			}), 
         ( 'pooling',   {	'poolsize':(2,2)
			}),
         ( 'hidden'  ,  {	'n_out':10
			}),
         ( 'logistic',  {	'n_out':2
			})
]
specs =  [
          
         ( 'hidden'  ,  {	'n_out':250
			}),
         ( 'dropout',   {	'prob':0.5
			}), 
         ( 'logistic',  {	'n_out':data.dim_out
			})
]


lr = 0.01
lr_exp = 0.5
L2_reg = 0.0
batch_size = 300
M = voo.models.Generic_model(n_in = data.dim, n_out = data.dim_out, data= data, 
                layerSpecs = specs, batch_size=batch_size, rng=None,    activation=voo.ReLU)

print "Model Magnitude:", voo.Magnitude(M)
 
reload(voo)
reload(voo.train)
trainer = voo.train.SGDTrainer(M, learning_rate=lr, L1_reg=0., L2_reg=L2_reg)
trainer.train_minibatches()
"""
param_values = hlv_aux.get_params(M)

while lr > 0.00005:
    print "============ LR %f" % (lr)
    #hlv_aux.set_params(M, param_values)
    #param_values = hlv_train.train_minibatches(M, min_epochs=20, max_epochs=200)
    param_values = Train_minibatches(M, min_epochs=50, max_epochs=1000, validation_frequency=39)
    lr = lr * lr_exp
"""
