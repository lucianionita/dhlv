import theano.tensor as T
import theano
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import numpy as np
import init 
import hooloovoo_aux as hlv_aux
print "DEPRECATED HOOLOOVOO REFERENCE IN TRAIN.PY : hlv_aux"
import quick as q
theano.config.on_unused_input="warn"
import time
import sys

class SGDTrainer:
    def __init__(self,     model,
                           min_epochs = 100,
                           max_epochs = 1000, 
                           patience_increase=2, 
                           improvement_threshold=0.995,
                           validation_frequency=None):
        # remember the parameters
        self.model = model
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.patience_increase = patience_increase
        self.improvement_threshold = improvement_threshold
        self.validation_frequency = validation_frequency
        if self.validation_frequency==None:
            self.validation_frequency=model.n_train_batches
        self.batch_size = self.model.batch_size
        # for ease of use
        self.n_train_batches = self.model.n_train_batches            
        
        self.best_params = hlv_aux.get_params(self.model)
        self.compile_model_ops()
    
    def compile_model_ops(self):
        self.model_ops = q.newObject()
        # Define the model training and testing functions for both
        # minibatches and arbitrary data
        index = T.lscalar()  # index to a [mini]batch
        x = T.matrix('x')
        y = T.ivector('y')  
        X = T.matrix('X')          
        Y = T.ivector('Y')     
        """
        self.model_ops.train_model = theano.function(inputs=[X, Y],
                                            outputs=self.model.errors(y),
                                            updates=self.model.updates,
                                            givens={
                                                x: X,
                                                y: Y})
                                                "
        self.model_ops.test_model2 = theano.function(  inputs=[],
                                            outputs=self.model.errors(y),
                                            givens={
                                                x: self.model.data.test.x,
                                                y: self.model.data.test.y})
                                                
        self.model_ops.validate_model2 = theano.function(inputs=[],
                                            outputs=self.model.errors(y),
                                            givens={
                                                x: self.model.data.valid.x,
                                                y: self.model.data.valid.y}) 
                                                """
        """                                                
        self.model_ops.minibatch = q.newObject()        
        self.model_ops.minibatch.test = theano.function(  inputs=[index],
                                            outputs=self.model.errors(y),
                                            givens={
                                                x: self.model.data.test.x[index * self.batch_size:(index + 1) * self.batch_size],
                                                y: self.model.data.test.y[index * self.batch_size:(index + 1) * self.batch_size]})
                                                
        self.model_ops.minibatch.validate = theano.function(inputs=[index],
                                            outputs=self.model.errors(y),
                                            givens={
                                                x: self.model.data.valid.x[index * self.batch_size:(index + 1) * self.batch_size],
                                                y: self.model.data.valid.y[index * self.batch_size:(index + 1) * self.batch_size]}) 
        self.model_ops.minibatch.train = theano.function(inputs=[index],
                                            outputs=self.model.cost,
                                            updates=self.updates,
                                            givens={
                                                x: self.model.data.train.x[index * self.batch_size:(index + 1) * self.batch_size],
                                                y: self.model.data.train.y[index * self.batch_size:(index + 1) * self.batch_size]})            
        """
        
    def train_minibatch(self, batch_idx, randomize = False):
        if randomize:
            self.model.randomize()
        return self.model.minibatch.train(batch_idx)
    
    def test_model(self):
        self.model.reset()
        errors = [self.model.minibatch.test(batch_idx) for batch_idx in range(self.model.n_test_batches)]
        return np.mean(errors)        

    def validate_model(self):
        self.model.reset()
        errors = [self.model.minibatch.validate(batch_idx) for batch_idx in range(self.model.n_test_batches)]
        return np.mean(errors)        
        
    def train_minibatches(self):
        # initialization parameters
        hlv_aux.set_params(self.model, self.best_params)
        best_validation_loss = np.inf
        test_score = 0.
        start_time = time.time()
        epoch = 0
        done_looping = False
        patience = self.model.n_train_batches * self.min_epochs
        
        # Ensure everything is all right with the model
        print "Before we start:"
        best_validation_loss = self.validate_model()    
        test_score = self.test_model()    
        print "     Validation score", best_validation_loss
        print "     Test       score", test_score
    
        while (not done_looping):
            t0 = time.time()

            for batch_idx  in xrange(self.n_train_batches):
                # Train on one batch
                batch_avg_cost = self.train_minibatch(batch_idx, randomize=True)
                iteration = epoch * self.n_train_batches + batch_idx
                t1 = time.time()
                sys.stdout.write("Training batch %i/%i, Time(elapsed/estimated) %.0fs/%.0fs                          \r" %(batch_idx+1, self.n_train_batches, t1-t0, (t1-t0)/(batch_idx+1)*self.n_train_batches))
                sys.stdout.flush()
                
                # validation if right time
                if (iteration + 1) % self.validation_frequency == 0:
                    # get validation loss
                    validation_loss = self.validate_model()
                    print('\nepoch %i, mb %i/%i, tcost %.5f, verror %.3f%%' % \
                        (epoch, batch_idx + 1, self.n_train_batches,
                         batch_avg_cost, validation_loss * 100.))
                    # Check if validation error is worth looking at
                    if validation_loss < best_validation_loss * self.improvement_threshold:                                                
                        patience = max(patience, iteration * self.patience_increase)
                        best_validation_loss = validation_loss
                        test_score = self.test_model()
                        print(('     epoch %i, minibatch %i/%i, test error of best'
                            ' model %0.3f%%') % (epoch, batch_idx + 1, 
                            self.n_train_batches, test_score * 100.))
                        self.best_params = hlv_aux.get_params(self.model)
            # check if done looping
            epoch = epoch + 1
            if epoch == self.max_epochs or patience <= iteration:
                done_looping = True
        end_time = time.time()
        print "Optimization Complete!"
        print "Best validation error %.3f%%." % (best_validation_loss * 100.)
        print "           Test Error: %.3f%%" % (test_score * 100.)
        print 'The training ran for %d epochs, with about %f epochs/sec' % ( epoch, 1. * epoch / (end_time - start_time))
        