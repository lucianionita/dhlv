import theano.tensor as T
import theano
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import numpy as np
import init 


class Layer(object):
    def __init__(self):
        pass
        self.options = []
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

class ConvLayer(Layer):
    def __init__(self, input, input_shape, batch_size, rng, n_filters, filter=(2,2), activation=T.tanh, poolsize=(2,2)):
        # input: the 4D input 
        # input_shape: the shape description : (channels, dim_x, dim_y)
        # batch_size: number of batches
        # n-filters = number of filter channels
        # pool_shape: the shape of the downsampling pool, used for initializing
        #               the weights
        # filter:   shape of the filter
        # initialize with Layer parent
        Layer.__init__(self)

        # standard stuff
        self.rng = rng
        self.input = input
        self.batch_size = batch_size

        # set shapes
        if len(input_shape)==2:
            input_shape = (1,) + input_shape
        
        self.input_shape = input_shape
        self.output_shape = (n_filters,input_shape[1]+1-filter[0],input_shape[2]+1-filter[1] )
        self.filter_shape = (n_filters,) + filter
        
        # this is used by the conv2d and has to be:
        #   (n_channels, n_channels_in, filter_dimx, filter_dimy)
        conv_filter_shape = (n_filters, input_shape[0]) + filter
        
        # this is used by conv2d and has to be:
        #   (batch_size, n_channels_in, img_dim_x, img_dim_y)
        conv_image_shape = (batch_size, input_shape[0]) + input_shape[1:]
        
        # initialize weights with random weights
        self.W = init.init_conv(rng, conv_filter_shape, poolsize)

        # the bias is a 1D tensor -- one bias per output feature map        
        self.b = init.init_zero((self.filter_shape[0],))

        # convolve input feature maps with filters
        
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=conv_filter_shape, 
                image_shape=conv_image_shape) # !!! OPTIMIZATable


        self.output = activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


class PoolingLayer(Layer):
    def __init__(self, input, input_shape, batch_size, rng, poolsize=(2, 2)):
        # initialize layer parent
        Layer.__init__(self)
        
        # set input
        self.input = input
        self.input_shape = input_shape
        self.output_shape = (input_shape[0], input_shape[1]/poolsize[0], input_shape[2]/poolsize[1])        
        
        self.poolsize = poolsize

        # downsample each feature map individually, using maxpooling
        self.output = downsample.max_pool_2d(input=self.input,
                                            ds=poolsize, ignore_border=True)
        # store parameters of this layer
        self.params = []

class NoiseLayer(Layer):
    def __init__(self, input, input_shape, batch_size, rng, scale=0.1):
        # initialize layer parent
        Layer.__init__(self)
        self.options.append('randomize-reset')        
        
        # set input
        self.input = input
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.rng = rng
        self.scale = scale

        self.noise = init.init_zero(self.output_shape)
        self.reset()
            
        self.output = self.input + self.noise
        self.params = []
    def randomize(self):
        self.noise = np.random.normal(0, scale, self.output_shape)
    def reset(self):
        self.noise = init.init_zero(self.output_shape)

class DropOutLayer(Layer):
    def __init__(self, input, input_shape, batch_size, rng, prob=0.5):
        # initialize layer parent
        Layer.__init__(self)
        self.options.append('randomize-reset')        
        
        # set input
        self.input = input
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.rng = rng
        self.prob = prob

        self.mask = init.init_zero(self.output_shape)
        self.reset()
            
        self.output = self.input * self.mask
        self.params = []
    def randomize(self):
        self.mask.set_value ( np.float32(self.rng.binomial(n=1, p=1-self.prob, size=self.output_shape)))
    def reset(self):
        self.mask.set_value ( np.ones(self.output_shape, dtype=theano.config.floatX) )

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
        
class FullyConnectedLayer_LowRank(Layer):
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
        self.output_shape = (n_out, )
    def cost(self, y):
        return negative_log_likelihood(self, y)
    def negative_log_likelihood(self,y):
        return -T.mean(T.log(self.output)[T.arange(y.shape[0]), y])
    def errors(self, y):
         return T.mean(T.neq(self.y_pred, y))


verbose = True

def GenLayer(layerClass, last_layer, batch_size, rng, config):
    input = last_layer.output
    input_shape = last_layer.output_shape
    
    # If we're doing a hidden layer, the input must be reshuffled
    if (layerClass == FullyConnectedLayer or \
                    layerClass == FullyConnectedLayer_LowRank or \
                    layerClass == LogisticRegressionLayer) and \
                    (len(input_shape)!=1):
        input = input.flatten(2)        
        input_shape = (np.prod(input_shape[0:]),)
    if (layerClass == ConvLayer):
        if len(input_shape)==2:
            input = input.reshape((batch_size, 1) + input_shape)
        
    
    
    new_layer = layerClass(    input = input, 
                               input_shape = input_shape,
                               batch_size = batch_size,
                               rng = rng, 
                               **config
                               )
    if verbose:
        print "New Layer ---------------------------------------------------"
        print "     Type:                   ", layerClass
        print "     No. parameters          ", np.sum([np.prod(p.eval().shape) for p in new_layer.params])
        print "     Prev. output shape:     ", last_layer.output_shape
        print "     Input Shape:            ", input_shape
        print "     Output shape:           ", new_layer.output_shape
    return new_layer
