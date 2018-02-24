from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import cPickle
import os
from scipy.io import loadmat

os.chdir(os.path.dirname(__file__))

def forward(x, W0, b0):
    '''
    x: N*784
    W0: 10*784
    b0: 10*1

    output: 10*N
    (each COLUMN is the probabilities of the image being 0..9)
    '''
    L0 = np.dot(W0, x.T) + b0
    output = softmax(L0)
    return output

def softmax(y):
    '''    
    y: 10*N
    
    output: 10*N
    '''
    return (exp(y)/tile(sum(exp(y),0), (len(y),1)))

