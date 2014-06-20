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

import cv2.cv as cv
import cv2
import numpy as np
import threading
import os
import cPickle as pickle
from matplotlib import pyplot as plt
import bz2, gzip
"""
### Stuff for computer vision
"""


class newObject(object): pass

lastKey = -1
visualization = True

def waitKey(milliseconds):
    global lastKey
    k = cv2.waitKey(milliseconds)
    if (k!=-1):
        lastKey = k
    return k        

def getLastKey():
    global lastKey
    k = lastKey
    lastKey = -1
    return k

def show(img, t=-1, w="quickshow", r=None, s=None):
    global visualization
    if (not visualization): return
    if not r==None:
        cv2.imshow(w, resizeFactor(img, r));
        waitKey(t)
    elif not s==None:
        cv2.imshow(w, resizeTo(img, s));
    else:
        cv2.imshow(w, img);
        waitKey(t)

def toHSV(img):
    return cv2.cvtColor(img, cv.CV_BGR2HSV)

def to0SV(img):
    x =  cv2.cvtColor(img, cv.CV_BGR2HSV)
    x[:,:,0]=0
    return x
    
def toLab(img):
    return cv2.cvtColor(img, cv.CV_BGR2Lab)

def toLuv(img):
    return cv2.cvtColor(img, cv.CV_BGR2Luv)
def toYCC(img):
    return cv2.cvtColor(img, cv.CV_BGR2YCrCb)

def toGRAY(img):
    return cv2.cvtColor(img, cv.CV_BGR2GRAY)
    
def toMaxGray(img):
    return np.maximum(np.maximum(img[:,:,0], img[:,:,1]), img[:,:,2])

def toNRGB(img):
    i = np.float32(img)
    ii = i[:,:,0] + i[:,:,1] + i[:,:,2]
    i[:,:,0] *= 255/ii
    i[:,:,1] *= 255/ii
    i[:,:,2] *= 255/ii
    return i/255

def fromHSV(img):
    return cv2.cvtColor(img, cv.CV_HSV2BGR)

def fromLab(img):
    return cv2.cvtColor(img, cv.CV_Lab2BGR)

def fromYCC(img):
    return cv2.cvtColor(img, cv.CV_YCrCb2BGR)

def fromGRAY(img):
    return cv2.cvtColor(img, cv.CV_GRAY2BGR)

def blur(img, ksize):
    return cv2.GaussianBlur(img, (ksize,ksize), ksize*0.33)

def Sobel(img, v, h, double=False):
    if double:
        return SobelX(img, v, h)
    if v==0 or h==0:
        return cv2.Sobel(img, cv2.cv.CV_8U, v, h)
    else:
        vv = cv2.Sobel(img, cv2.cv.CV_8U, v, 0)
        hh = cv2.Sobel(img, cv2.cv.CV_8U, 0, h)
        return np.maximum(vv,hh)

def SobelX(img, v, h):
    a = Sobel(img, v, h)
    b = Sobel(img[::-1, ::-1],v,h)[::-1, ::-1]
    return np.maximum(a,b)

def getCircleKernel(n=5):
    if (type(n) == np.int): # int        
        m = (n-1)/2
        a = (np.cumsum(np.ones((n,n)), axis=0)-m-1)**2
        b = (np.cumsum(np.ones((n,n)), axis=1)-m-1)**2
        return (a + b <= m * m).astype(np.uint8)
    else: #tuple
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,n)
        
    
def getSpecular(img):
    img_hsv = toHSV(img)
    img_h = cv2.inRange(img_hsv, (0,0,224),(255,64,255))
    img_h = cv2.dilate(img_h, ())        
    img_h = cv2.erode(img_h, ())
    img_h = cv2.erode(img_h, ())
    img_h = cv2.dilate(img_h, ())
    return img_h

def initCapture():
    global capture
    capture = 0
    capture = cv.CaptureFromCAM(capture)
    
def getImgCAM(flushframes = 10):
    global capture
    for i in range(flushframes):
        cv.GrabFrame(capture)
        img = cv.RetrieveFrame(capture)
    # convert to numpy format
    img = np.fliplr(np.asarray(img[:,:]))
    return img
   
def resizeFactor(img, factor, inter = cv2.cv.CV_INTER_CUBIC):
    w = int(img.shape[0] * factor)
    h = int(img.shape[1] * factor)
    return cv2.resize(img, (h,w), interpolation = inter)

def resizeTo(img, dim, inter = cv2.cv.CV_INTER_CUBIC):
    return cv2.resize(img, dim, interpolation = inter)
    
    
    
"""
Face detector helper
"""

fd = cv2.CascadeClassifier("/home/tc/Downloads/opencv-2.4.8/data/haarcascades/haarcascade_frontalface_alt.xml")

def rect2corners(r):
    return r[0], r[1], r[0]+r[2], r[1]+r[3]
    
    
"""
### Stuff for threading
"""

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

def quickCalibrateCamera(nframes=100):
    print "Calibrating Camera"
    os.system('uvcdynctrl -s "White Balance Temperature, Auto" 1')
    os.system('uvcdynctrl -s "Exposure, Auto" 3')
    os.system('uvcdynctrl -s "Focus, Auto" 1')
    for i in range(100):
        getImgCAM(1)
    os.system('uvcdynctrl -s "White Balance Temperature, Auto" 0')
    os.system('uvcdynctrl -s "Exposure, Auto" 1')
    os.system('uvcdynctrl -s "Focus, Auto" 0')
    print "done"

cameraCalibration23 = []

def doManualCalibration():
    global cameraCalibration23
    cameraCalibration23 = CalibrateCameraManually()
def DilDelEro(img, n=7):
    k = getCircleKernel(n)
    a = cv2.erode(img, k)
    b = cv2.dilate(img, k)
    return b-a

def EroDil(img, n=7):
    k = getCircleKernel(n)
    a = cv2.erode(img, k)
    b = cv2.dilate(a, k)
    return b

def DilEro(img, n=7):
    k = getCircleKernel(n)
    b = cv2.dilate(img, k)
    a = cv2.erode(b, k)
    return a
    
def normalize01(x):
    a = np.min(x)
    b = np.max(x)
    return (x-a)/(b-a)

def reCalibrateCamera():
    global cameraCalibration23
    for i in cameraCalibration23:
        os.system(i)
        print i

def getGaussian2D(ksize, sigma):
    g = cv2.getGaussianKernel(ksize,sigma)
    return g * np.transpose(g)

def getCircleKernel(n=5):
    m = (n-1)/2
    a = (np.cumsum(np.ones((n,n)), axis=0)-m-1)**2
    b = (np.cumsum(np.ones((n,n)), axis=1)-m-1)**2
    return (a + b <= m * m).astype(np.uint8)


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
