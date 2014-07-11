import numpy as np


ReLU = lambda x: T.maximum(0, x)

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
    return np.log(tsize) / np.log(2)
         
                    
