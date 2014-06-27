import theano.tensor as T
import theano
import numpy as np
from .. import init 
import decomp

class Layer(object):
    def __init__(self):
        pass
    def output_shape(self):
        raise NotImplementedError
    def output(self):
        raise NotImplementedError

class InputLayer(Layer):
    def __init__(self, input, input_shape, batch_size):
        Layer.__init__(self)
        self.input = input
        self.input_shape = input_shape
        self.output = self.input
        self.output_shape = self.input_shape
        self.params = []


class FullyConnectedLayer(Layer):
    def __init__(self,  input, input_shape, batch_size, rng, n_out, activation=T.tanh):
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
        W = init.init_standard(rng, (n_in, n_out))
        if activation == theano.tensor.nnet.sigmoid:
            W.set_value(W.get_value()*4)
        b = init.init_zero((n_out,))
        self.W = W
        self.b = b

        #define output        
        lin_output = T.dot(input, self.W) + self.b
        self.output = activation(lin_output)
        
        # define params
        self.params = [self.W, self.b]


        
    
class LogisticRegressionLayer(Layer):
    def __init__(self, input, input_shape, batch_size, rng, n_out, activation=T.tanh):
        # initialize with Layer parent
        Layer.__init__(self)
        
        # standard stuff
        self.rng = rng
        self.input = input
        self.batch_size = batch_size
        
        # define model
        n_in = input_shape[0]
        self.W = init.init_zero((n_in,n_out))
        self.b = init.init_zero((n_out,))

        # define output
        self.output = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.output, axis=1)
        self.params = [self.W, self.b]
    def cost(self, y):
        return negative_log_likelihood(self, y)
    def negative_log_likelihood(self,y):
        return -T.mean(T.log(self.output)[T.arange(y.shape[0]), y])
    def errors(self, y):
         return T.mean(T.neq(self.y_pred, y))


def GenLayer(layerClass, last_layer, batch_size, rng, config):
    input = last_layer.output
    input_shape = last_layer.output_shape
    
    # If we're doing a hidden layer, the input must be reshuffled
    if (layerClass == FullyConnectedLayer or \
                    layerClass==LogisticRegressionLayer) and \
                    (len(input_shape)!=1):
        input = input.flatten(2)
        input_shape = (input_shape[0], np.prod(input_shape[1:]))
    
    new_layer = layerClass(    input = input, 
                               input_shape = input_shape,
                               batch_size = batch_size,
                               rng = rng, 
                               **config
                               )
    return new_layer