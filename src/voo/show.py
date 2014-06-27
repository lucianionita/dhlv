# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 18:08:14 2014

@author: tc
"""

def show_filter(x, r=1, size=64):
    q.show((x.reshape(size,size)-np.min(x)) / (np.max(x) - np.min(x)), r=r)

def show_filter2(x, r=1, size=64):
    q.show((x.reshape(size,size)) / (np.max(np.abs(x)))+0.5, r=r)

def show_filters(X, r=1, size=64):
    for i in range (X.shape[1]):
        show_filter(param_values[0][:,i], r, size)

def show_filters2(X, r=1, size=64):
    for i in range (X.shape[1]):
        show_filter2(param_values[0][:,i], r, size)

def showBunch(X, r=1, size=64):
    n = X.size/size/size
    s = size
    for i in range(732,733):
        Y = X.copy()
        if i & 16: Y = Y.reshape(s,s,n)
        if i & 64: Y = Y.transpose()
        if i & 128: Y = Y.reshape(n*s,s)
        if i & 512: Y = Y.transpose()
        Y = Y.reshape(s,s*n)        
        print i
        q.show(Y/(np.max(Y))+0.5, r=r, t=-1)
