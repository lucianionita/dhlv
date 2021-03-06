import os
import cPickle as pickle
import quick as q
import theano
import sys
import time
import numpy as np
import warnings
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import voo


if 'data_faces' in dir(): 
    print "... Data seems to be already loaded"
else:
    data = voo.datasets.get_mnist()

"""
         ( 'conv'  ,    {	'n_filters':32,
				'filter':(7,7)
			}), 
         ( 'pooling',   {	'poolsize':(3,3)	
			}),
         ( 'dropout',   {       'prob':0.5
                        }),
         ( 'conv'  ,    {	'n_filters':64,     
				'filter':(4,4)
			}), 
         ( 'pooling',   {	'poolsize':(2,2)	
			}),
         ( 'dropout',   {       'prob':0.5
                        }),
         ( 'conv'  ,    {	'n_filters':128,
				'filter':(2,2)
			}), 
         ( 'pooling',   {	'poolsize':(2,2)
			}),
         ( 'dropout',   {       'prob':0.5
                        }),
         ( 'hidden',    {	'n_out':1000
			}), 
         ( 'dropout',   {       'prob':0.5
                        }),
         ( 'hidden',    {	'n_out':250
			}), 
         ( 'dropout',   {       'prob':0.5
                        }),
         ( 'logistic',  {	'n_out':10
			})
"""
specs = [
#         ( 'noise',   {       'scale':0.25
#                        }),
#         ( 'dropout',   {       'prob':0.5
#                        }),
         ( 'conv'  ,    {	'n_filters':32,
				'filter':(7,7)
			}), 
         ( 'pooling',   {	'poolsize':(3,3)	
			}),
         ( 'dropout',   {       'prob':0.5
                        }),
         ( 'hidden',    {	'n_out':100
			}), 
         ( 'logistic',  {	'n_out':10
			})
]
lr = 0.02
lr_exp = 0.1
L2_reg = 0.0
batch_size = 600
M = voo.models.Generic_model(n_in = data.dim, n_out = data.dim_out, data= data, 
                layerSpecs = specs, batch_size=batch_size, rng=None, activation=voo.ReLU)
	
print "Model Magnitude: %.3f" % voo.Magnitude(M)
 
trainer = voo.train.SGDTrainer(M, learning_rate=lr, L1_reg=0., L2_reg=L2_reg, max_epochs=10000, min_epochs=250, min_epochs_retain=100)
best_params = voo.get_params(M)
while lr > 0.0001:
        print "Learning Rate:", lr
	voo.set_params(M, best_params)
	trainer.train_minibatches()
	best_params = voo.get_params(M)
	lr = lr * lr_exp
	trainer.lr.set_value(lr)
