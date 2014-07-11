print "Voo library. Version 0.135"
#__all__ = ["layers"]
import layers
import init
import train
import models
import datasets
import todo

from misc import *
import theano
theano.config.openmp = True
