import theano.tensor as T
import theano
import numpy as np
from . import  init
import voo

# !!! What do I have to do to import Layer???
class Layer(object):
    def __init__(self):
        pass
    def output_shape(self):
        raise NotImplementedError
    def output(self):
        raise NotImplementedError

class FullyConnectedDecompLayer(Layer):
    def __init__(self,  input, input_shape, batch_size, rng, n_out, activation=T.tanh, dim=10):
        # initialize with Layer parent
        Layer.__init__(self)
        
        # standard stuff
        self.rng = rng
        self.input = input
        self.batch_size = batch_size
        
        # set shapes
        self.input_shape = input_shape
        self.output_shape = (n_out,)
 
        # define the model parameters
        n_in = np.prod(input_shape)
        self.W_l = init.init_standard(rng, (n_in, dim))
        self.W_r = init.init_standard(rng, (dim, n_out))
        if activation == theano.tensor.nnet.sigmoid:
            W_l.set_value(W_l.get_value()*4)
            W_r.set_value(W_r.get_value()*4)
        b = init.init_zero((n_out,))
        self.W = T.dot(self.W_l, self.W_r)
        self.b = b

        #define output        
        lin_output = T.dot(input, self.W) + self.b
        self.output = activation(lin_output)
        
        # define params
        self.params = [self.W_l, self.W_r, self.b]