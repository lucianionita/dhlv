import numpy as np
import theano

def init_standard(rng, shape):
        X_bound = np.sqrt(6. / np.sum(shape))
        X = theano.shared(np.asarray(
            rng.uniform(low=-X_bound, high=X_bound, size=shape),
            dtype=theano.config.floatX),  borrow=True)
        return X

def init_conv(rng, shape, poolsize):
        fan_in = np.prod(shape[1:])
        fan_out = (shape[0] * np.prod(shape[2:]) /np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        X_bound = np.sqrt(6. / (fan_in + fan_out))
        X = theano.shared(np.asarray(
            rng.uniform(low=-X_bound, high=X_bound, size=shape),
            dtype=theano.config.floatX),  borrow=True)
        return X


def init_zero(shape):
        values = np.zeros(shape, dtype=theano.config.floatX)
        X = theano.shared(value=values, borrow=True)
        return X
