# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:29:43 2013

@author: tc

Library of quick  ols that should save up some time
The goal is to just have import quick as q, and then
use the module for quick things. img_hsv = q.bgr2hsv(img)
etc. Also, more particular functions that do end up here should 
be moved to their own module.
"""

import numpy as np
import threading
import os
import cPickle as pickle
import bz2, gzip
"""
### Stuff for computer vision
"""


class newObject(object): pass

def runMethodThread(func, args, queue):
    queue.append(func(*args))

class myThread:
    def __init__(self, func, args):
        self.func = func
        self.args = args
        self.queue = []
        self.thread = threading.Thread(target=runMethodThread, args=(func, args, self.queue))
    def start(self):
        self.thread.start()
    def join(self):
        self.thread.join()
    def stop(self):
        self.thread._Thread__stop()
    def isrunning(self):
        return self.thread.isAlive()
    def result(self):
        return self.queue.pop()

# Pickle stuff

def load_from_pkl(filename, compression="bz2"):
    if compression=="bz2":
        f = bz2.BZ2File(filename, 'r')
    elif compression=="gz":
        f = gzip.GzipFile(filename, 'r')
    else:
        f = open(filename, 'r')
    x = pickle.load(f)
    f.close()
    return x

def save_to_pkl(filename, x, compression="bz2"):
    if compression=="bz2":
        f = bz2.BZ2File(filename, 'w')
    elif compression=="gz":
        f = gzip.GzipFile(filename, 'w')
    else:
        f = open(filename, 'w')
    #f =  open(str(t)+".pbz2", 'wb');
    pickle.dump(x, f)
    f.close()
    #hg.stop("save2file")
