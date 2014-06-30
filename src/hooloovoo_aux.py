import os
import cPickle as pickle
import cPickle
import quick as q
import numpy as np
import time
import sys
import time
import gzip
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
# Auxiliary files

# The theano tutorial load_data, used to load the MNIST data
def mnist_data(dataset='mnist.pkl.gz'):
    t0 = time.time()
    print "Loading digits data ..."
    dataset = "/home/ubuntu/dhlv/data/"+dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    def shared_dataset(data_xy, borrow=True):

        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                    dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                    dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, theano.config.floatX)

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    data_digits = q.newObject()
    data_digits.train = q.newObject()
    data_digits.valid = q.newObject()
    data_digits.test = q.newObject()
    
    data_digits.train.x = train_set_x
    data_digits.valid.x = valid_set_x
    data_digits.test.x  = test_set_x
    data_digits.train.y = train_set_y
    data_digits.valid.y = valid_set_y
    data_digits.test.y  = test_set_y
    t1 = time.time()
    print "Data loaded in %0.2f seconds" % ( t1-t0)
    return data_digits



# Get the data
def faces_data(resize_factor = None):
    print "Loading faces data ..."
    t0 = time.time()
    faces, labels = q.load_from_pkl("/home/ubuntu/dhlv/data/faces.bzpkl")
    if resize_factor is not None:
        new_size = (int(64*resize_factor), int(64*resize_factor))
        print new_size
        for i in range(len(faces)):
            print i
            #t = cv2.resize(faces[i].reshape(64,64),new_size).ravel()
    labels = np.asarray(labels, dtype=np.int32)
    faces = np.asarray(faces, dtype=np.float32)
    
    
    data = q.newObject()
    data.train = q.newObject()
    data.valid = q.newObject()
    data.test = q.newObject()
    
    mean = 128
    stdev = 75
    
    faces = (faces - mean) / stdev
    np.random.seed(10)
    np.random.shuffle(faces)
    np.random.seed(10)
    np.random.shuffle(labels)
    
    data.train.x = np.vstack((faces[labels==1,:][    :1500], faces[labels==-1,:][    :1500]))
    data.valid.x = np.vstack((faces[labels==1,:][1500:2000], faces[labels==-1,:][1500:2000]))
    data.test.x =  np.vstack((faces[labels==1,:][2000:2400], faces[labels==-1,:][2000:2400]))
    
    
    
    data.train.y = np.hstack((np.ones((1500, ), dtype=np.int32),
                              np.zeros((1500, ),dtype=np.int32)))
    data.valid.y = np.hstack((np.ones((500, ),dtype=np.int32),
                              np.zeros((500, ),dtype=np.int32)))
    data.test.y  = np.hstack((np.ones((400, ),dtype=np.int32),
                              np.zeros((400, ),dtype=np.int32)))
    


    data.train.x = theano.shared(data.train.x, borrow=True)
    data.train.y = theano.shared(data.train.y, borrow=True)
    data.valid.x = theano.shared(data.valid.x, borrow=True)
    data.valid.y = theano.shared(data.valid.y, borrow=True)
    data.test.x  = theano.shared(data.test.x , borrow=True)
    data.test.y  = theano.shared(data.test.y , borrow=True)
    
    
    def show(img):
        q.show(img.reshape((64,64)))
        cv2.destroyAllWindows()
        for k in range(10):
            cv2.waitKey(10)
    data.show = show
    t1 = time.time()
    print "Data loaded in %0.2f seconds" % ( t1-t0)
    return data
        



def get_params(model):
    param_values = []
    for param in model.params:
        param_values.append(param.get_value().copy());
    return param_values;

def set_params(model, param_values):
    for param, value in zip(model.params, param_values):
        param.set_value(value)        

def Magnitude(model):
    psize = [np.prod(param.eval().shape) for param in model.params]
    tsize = np.sum(psize)
    return round(np.log(tsize) / np.log(2),)
         
                    
