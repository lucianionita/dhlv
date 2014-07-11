import theano.tensor as T
import theano
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import numpy as np
import init 
import layers
import quick as q

class Generic_model():
    def __init__(self, n_in, n_out, data, layerSpecs, batch_size, rng, activation=T.tanh):
	print "Generating model with these specs:", layerSpecs

        # Define classifier independent stuff
        # preliminaries
        #######################################################################

        index = T.lscalar()  # index to a [mini]batch
        x = T.matrix('x')
        y = T.ivector('y')  
        X = T.matrix('X')  
        Y = T.ivector('Y')       
        
        self.index = index
        self.x = x
        self.y = y
        self.X = X
        self.Y = Y
        
        if (rng is None):
            rng = np.random.RandomState(23455)  
            
        self.batch_size = batch_size
        self.n_train_batches = data.train.x.get_value(borrow=True).shape[0] / batch_size
        self.n_valid_batches = data.valid.x.get_value(borrow=True).shape[0] / batch_size
        self.n_test_batches  = data.test.x.get_value (borrow=True).shape[0] / batch_size        
        
        self.data = data 
        
        
        # Define the model structure    
        #######################################################################
        
        layerClasses = {
            'conv': layers.ConvLayer,
            'hidden': layers.FullyConnectedLayer,
            'hidden_lr': layers.FullyConnectedLayer_LowRank,
            'logistic': layers.LogisticRegressionLayer,
            'pooling': layers.PoolingLayer,
            'dropout': layers.DropOutLayer
            }
        
        Layers = []

        input_layer = layers.InputLayer(x, n_in, batch_size)
        Layers.append(input_layer)
        prev_layer = input_layer
        for layer_idx in range(len(layerSpecs)):
            # First we get the layer class and specs
            layerType, layerConfig = layerSpecs[layer_idx]
            layerClass = layerClasses[layerType]           
            new_layer = layers.GenLayer(layerClass, prev_layer, batch_size, rng, layerConfig)
            
            Layers.append(new_layer)
            prev_layer = new_layer
        
        self.Layers=Layers

        #
        # Define all the model specifics that relate to the parameters:
        # vector of parameters, gradients, errors, updates, accuracy cost
        # regularization cost, overall cost
        ######################################################################


        self.params = []
        for layer in Layers:
            self.params.extend(layer.params)
        

        self.acc_cost = self.Layers[-1].negative_log_likelihood(y)
        #self.L1 = np.sum([abs(param).sum() for param in self.params])
        #self.L2 = np.sum([(param**2).sum() for param in self.params])
        #self.reg_cost = L1_reg * self.L1 + L2_reg * self.L2
        #self.cost = self.acc_cost + self.reg_cost


        """
        self.grads = [T.grad(cost=self.cost, wrt=param) for param in self.params]    
        
        self.updates = [(param, param - self.lr * grad) for
                    param, grad in zip(self.params, self.grads)]
        self.minibatch = q.newObject()        
        self.minibatch.test = theano.function(  inputs=[index],
                                            outputs=self.errors(y),
                                            givens={
                                                x: self.data.test.x[index * self.batch_size:(index + 1) * self.batch_size],
                                                y: self.data.test.y[index * self.batch_size:(index + 1) * self.batch_size]})
                                                
        self.minibatch.validate = theano.function(inputs=[index],
                                            outputs=self.errors(y),
                                            givens={
                                                x: self.data.valid.x[index * self.batch_size:(index + 1) * self.batch_size],
                                                y: self.data.valid.y[index * self.batch_size:(index + 1) * self.batch_size]}) 
        self.minibatch.train = theano.function(inputs=[index],
                                            outputs=self.cost,
                                            updates=self.updates,
                                            givens={
                                                x: self.data.train.x[index * self.batch_size:(index + 1) * self.batch_size],
                                                y: self.data.train.y[index * self.batch_size:(index + 1) * self.batch_size]})            

        """           
        """
        def validate_model(self):
            validation_losses = [self.minibatch.validate(i) for i
                                         in xrange(self.n_valid_batches)]
            return np.mean(validation_losses)
        def test_model(self):
            test_losses = [self.minibatch.test(i) for i
                                         in xrange(self.n_test_batches)]
            return np.mean(test_losses) 
        """
    def errors(self, y):
        return self.Layers[-1].errors(y)
    
    def randomize(self):
        for layer in self.Layers:
            if 'randomize-reset' in layer.options:
                layer.randomize()
    def reset(self):
        for layer in self.Layers:
            if 'randomize-reset' in layer.options:
                layer.reset()
