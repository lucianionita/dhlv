import numpy as np
import theano

def init_standard(rng, shape, total_dim):
        X_bound = np.sqrt(6. / total_dim)
        X = theano.shared(np.asarray(
            rng.uniform(low=-X_bound, high=X_bound, size=shape),
            dtype=theano.config.floatX),  borrow=True)
        return X

def init_zero(shape):
        values = np.zeros(shape, dtype=theano.config.floatX)
        X = theano.shared(value=values, borrow=True)
        return X