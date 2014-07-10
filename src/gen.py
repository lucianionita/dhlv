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
    data = voo.datasets.get_faces()
"""    
if 'data_digits' in dir(): 
    print "... Data seems to be already loaded"
else:
    data = voo.datasets.get_mnist()
"""


specs = [
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
         ( 'logistic',  {	'n_out':2
			})
]
lr = 0.1
lr_exp = 0.5
L2_reg = 0.0
batch_size = 500
M = voo.models.Generic_model(n_in = data.dim, n_out = data.dim_out, data= data, 
                layerSpecs = specs, batch_size=batch_size, rng=None,    activation=voo.ReLU)

print "Model Magnitude: %.3f" % voo.Magnitude(M)
 
trainer = voo.train.SGDTrainer(M, learning_rate=lr, L1_reg=0., L2_reg=L2_reg, max_epochs=2000)
trainer.train_minibatches()
trainer.lr.set_value(0.005)
trainer.train_minibatches()
