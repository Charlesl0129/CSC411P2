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
def get_test(M):
    batch_xs = np.zeros((0, 28*28))
    batch_y_s = np.zeros( (0, 10))
    
    test_k =  ["test"+str(i) for i in range(10)]
    for k in range(10):
        batch_xs = np.vstack((batch_xs, ((np.array(M[test_k[k]])[:])/255.)  ))
        one_hot = np.zeros(10)
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s,   np.tile(one_hot, (len(M[test_k[k]]), 1))   ))
    return batch_xs, batch_y_s

def get_train(M):
    batch_xs = np.zeros((0, 28*28))
    batch_y_s = np.zeros( (0, 10))
    
    train_k =  ["train"+str(i) for i in range(10)]
    for k in range(10):
        batch_xs = np.vstack((batch_xs, ((np.array(M[train_k[k]])[:])/255.)  ))
        one_hot = np.zeros(10)
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s,   np.tile(one_hot, (len(M[train_k[k]]), 1))   ))
    return batch_xs, batch_y_s

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

def grad_loss_W (Y,x,W0,b0):
    '''
    Y: 10:N
    x: N*784
    W0: 10*784
    b0:10*1

    output: 10*784
    '''
    output = forward(x,W0,b0)
    dy = output - Y

    return np.dot(dy,x)

def grad_loss_b (Y,x,W0,b0):
    '''
    Y: 10*N
    x: N*784
    W0: 10*784
    b0:10*1

    output: 10*1
    '''
    output = forward(x,W0,b0)
    dy = output - Y
    
    temp = np.ones((Y.shape[1],1))
    return np.dot(dy,temp)

def NLL(Y, output):
    '''
    Y: 10:N

    output: 10:N
    '''
    return -sum(Y*log(output)) 

def correctness (x,y,W0,H,h,b0):
    diff = NLL(y,forward(x, W0+H, b0))-NLL(y,forward(x, W0, b0))
    return diff/h

def df_finite_differenceW(x,y,W0,b0):
    h = 0.0001
    W0_change = W0.copy()
    row,column = shape(W0_change)
    for i in range(0,row):
        for j in range(0,column):
            W0_forward = W0.copy()
            W0_forward[i][j] = W0[i][j] + h
            W0_change[i][j] = (NLL(y,forward(x,W0_forward,b0)) - NLL(y,forward(x,W0,b0)))/h
    return W0_change
    
def df_finite_differenceb(x,y,W0,b0):
    h = 0.0001
    b0_change = b0.copy()
    row,column = shape(b0_change)
    for i in range(0,row):
        for j in range(0,column):
            b0_forward = b0.copy()
            b0_forward[i][j] = b0[i][j] + h
            b0_change[i][j] = (NLL(y,forward(x,W0,b0_forward)) - NLL(y,forward(x,W0,b0)))/h
    return b0_change
    

M = loadmat('mnist_all.mat')
train_x, train_y = get_train(M)
test_x, test_y = get_test(M)

dim_x = 28*28
dim_h = 20
dim_out = 10
numOutput = 10
numFeatures = 784
numTraining = 70
np.random.seed(0)
h = 0.001
W0 = np.random.normal(0.,0.5,[10,784])
b0 = np.random.normal(0.,0.5,[10,1])
part3y = train_y[0:1000]
part3x = train_x[0:1000]
gdW = grad_loss_W(part3y.T,part3x,W0,b0)
fdW = df_finite_differenceW(part3x,part3y.T,W0,b0)
gdb = grad_loss_b(part3y.T,part3x,W0,b0)
fdb = df_finite_differenceb(part3x,part3y.T,W0,b0)
errorW = []
errorb = []
countW = 0
countb = 0
for i in range (10):
    if (gdb+fdb)[i] != 0:
        errorb.append(abs((gdb-fdb)[i]/(gdb+fdb)[i]))
        countb += 1
    for j in range(28*28):
        if (gdW+fdW)[i][j] != 0 and abs((gdW-fdW)[i][j]/(gdW+fdW)[i][j])<0.99:
            errorW.append(abs((gdW-fdW)[i][j]/(gdW+fdW)[i][j]))
            countW += 1
perc_mean_error_W = np.mean(errorW)
perc_mean_error_b = np.mean(errorb)
perc_std_error_W = np.std(errorW)
perc_std_error_b = np.std(errorb)
print ('The percent difference of W is:', perc_mean_error_W*100,'(mean)',perc_std_error_W,'(standard deviation)','%')
print ('The percent difference of b is:', perc_mean_error_b*100,'(mean)',perc_std_error_b,'(standard deviation)','%')