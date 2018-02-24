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
np.random.seed(19)

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

def grad_descent(f_forward, NLL, dfdW, dfdb, x, y, x_t, y_t,init_W, init_b, alpha, max_iter):

    iterList = []
    perfTrainList = []
    perfTestList = []

    EPS = 1e-5   #EPS = 10**(-5)
    prev_W = init_W-10*EPS
    prev_b = init_b-10*EPS
    W = init_W.copy()
    b = init_b.copy()
    
    iter  = 1
    while norm(W - prev_W) + norm(b - prev_b) >  EPS and iter <= max_iter:
        prev_W = W.copy()
        prev_b = b.copy()
        W -= alpha*dfdW(y,x,W,b)
        b -= alpha*dfdb(y,x,W,b)
        if iter % 5 == 0:
            iterList.append (iter)
            #calculating performances:
            output = f_forward(x,W,b)
            count = 0.0
            yT = y.T
            outputT = output.T
            N = outputT.shape[0]
            for i in range (N):
                if argmax(yT[i]) == argmax(outputT[i]):
                    count += 1.0
            perfTrain = count / N
            perfTrainList.append(perfTrain)

            outputt = f_forward(x_t,W,b)
            count = 0.0
            ytT = y_t.T
            outputtT = outputt.T
            N = outputtT.shape[0]
            for i in range (N):
                if argmax(ytT[i]) == argmax(outputtT[i]):
                    count += 1.0
            perfTest = count / N
            perfTestList.append(perfTest)
             


            print "Iter", iter
            print "NLL", NLL(y,f_forward(x,W,b))
            print '\n'

        iter += 1
    return W, b, perfTrainList, perfTestList, iterList

def grad_descent_momentum(f_forward, NLL, dfdW, dfdb, x, y, x_t, y_t,init_W, init_b, alpha, max_iter):

    iterList = []
    perfTrainList = []
    perfTestList = []

    EPS = 1e-5   #EPS = 10**(-5)
    prev_W = init_W-10*EPS
    prev_b = init_b-10*EPS
    W = init_W.copy()
    b = init_b.copy()
    prev_vW = 0.0
    prev_vb = 0.0

    prev_loss =  NLL(y,f_forward(x,W,b))

    iter  = 1
    while norm(W - prev_W) + norm(b - prev_b) >  EPS and iter <= max_iter:
        prev_W = W.copy()
        prev_b = b.copy()
        vW = prev_vW * 0.9 + alpha * dfdW(y,x,W,b)
        W -= vW
        vb = prev_vb * 0.9 + alpha * dfdb(y,x,W,b)
        b -= vb        
        prev_vb = vb
        prev_vW = vW
        if iter % 50 == 0:
            iterList.append (iter)
            #calculating performances:
            output = f_forward(x,W,b)
            count = 0.0
            yT = y.T
            outputT = output.T
            N = outputT.shape[0]
            for i in range (N):
                if argmax(yT[i]) == argmax(outputT[i]):
                    count += 1.0
            perfTrain = count / N
            perfTrainList.append(perfTrain)

            outputt = f_forward(x_t,W,b)
            count = 0.0
            ytT = y_t.T
            outputtT = outputt.T
            N = outputtT.shape[0]
            for i in range (N):
                if argmax(ytT[i]) == argmax(outputtT[i]):
                    count += 1.0
            perfTest = count / N
            perfTestList.append(perfTest)
             
            print "Iter", iter
            print "NLL", NLL(y,f_forward(x,W,b))
            print '\n'
            
        iter += 1
    return W, b, perfTrainList, perfTestList, iterList

def part6grad_descent_momentum(f_forward, NLL, dfdW, dfdb, x, y, x_t, y_t,init_W, b, alpha, max_iter,adaptive ,outputi1,pixelj1,outputi2,pixelj2):

    mo_traj = []
    LossList = []

    #---------------------------
    W_adj = np.zeros([10,784])
    W_adj[outputi1][pixelj1] = 1
    W_adj[outputi2][pixelj2] = 1
    #---------------------------

    EPS = 1e-7   #EPS = 10**(-7)
    prev_W = init_W-10*EPS
    W = init_W.copy()
    prev_vW = 0.0
    iter  = 1
    while norm(W - prev_W) >  EPS and iter <= max_iter:
        mo_traj.append(((W*W_adj)[outputi1][pixelj1],(W*W_adj)[outputi2][pixelj2]))
        prev_W = W.copy()
        vW = prev_vW * 0.4 * W_adj + alpha * dfdW(y,x,W,b) * W_adj
        W -= vW
        prev_vW = vW.copy()
        #fill in contour plot data recording
        LossList.append (NLL(y,f_forward(x,W,b)))
        iter += 1
    return mo_traj, LossList
    
def part6grad_descent_no_momentum(f_forward, NLL, dfdW, dfdb, x, y, x_t, y_t,init_W, b, alpha, max_iter,adaptive ,outputi1,pixelj1,outputi2,pixelj2):

    gd_traj = []
    LossList = []

    #---------------------------
    W_adj = np.zeros([10,784])
    W_adj[outputi1][pixelj1] = 1
    W_adj[outputi2][pixelj2] = 1  
    #---------------------------

    EPS = 1e-7   #EPS = 10**(-7)
    prev_W = init_W-10*EPS
    W = init_W.copy()
    iter  = 1
    while norm(W - prev_W) >  EPS and iter <= max_iter:
        gd_traj.append(((W*W_adj)[outputi1][pixelj1],(W*W_adj)[outputi2][pixelj2]))        
        prev_W = W.copy()
        W -= alpha * dfdW(y,x,W,b) * W_adj
        #fill in contour plot data recording
        LossList.append (NLL(y,f_forward(x,W,b)))
        iter += 1
    return gd_traj, LossList

def getLoss(W0,b, w1,w2,Y,x,outputi1,pixelj1,outputi2,pixelj2):
    W = W0.copy()
    W[outputi1][pixelj1]=w1
    W[outputi2][pixelj2]=w2
    return NLL(Y, forward(x, W, b))
#-------------------------------------------------------------------------------------------------------------------------------------------------------

M = loadmat('mnist_all.mat')
train_x, train_y = get_train(M)
test_x, test_y = get_test(M)
    
dim_x = 28*28
dim_h = 20
dim_out = 10
numOutput = 10
numFeatures = 784
directory = 'storage'
if not os.path.exists(directory):
    os.makedirs(directory)
    W_init = np.random.normal(0.,0.5,[10,784])
    b_init = np.random.normal(0.,0.5,[10,1])
    
    W_trained, b_trained, perfTrainList, perfTestList, iterList = grad_descent_momentum (forward,NLL,grad_loss_W,grad_loss_b,train_x,train_y.T,test_x,test_y.T,W_init,b_init,0.000001,5000)
    np.save('storage/W trained.npy',W_trained)
    np.save('storage/b trained.npy',b_trained)

W = np.load('storage/W trained.npy')
b = np.load('storage/b trained.npy')

w1_c = 352
w2_c = 348
w1_r = 3  #change this to 4 for part e (not demonstrating good of momentum)
w2_r = 1

W[w1_r][w1_c] -= 2 #change to 0 to produce part a
W[w2_r][w2_c] += 2 #change to 0 to produce part a

#with momentum:
mo_traj, LossList = part6grad_descent_momentum(forward, NLL, grad_loss_W, grad_loss_b, train_x, train_y.T, test_x, test_y, W, b, 0.0035, 20,1 ,w1_r,w1_c,w2_r,w2_c)

#without momentum:
gd_traj, LossList = part6grad_descent_no_momentum(forward, NLL, grad_loss_W, grad_loss_b, train_x, train_y.T, test_x, test_y, W, b, 0.0035, 20,1 ,w1_r,w1_c,w2_r,w2_c)

w1s = np.arange(-2, 2, 0.1)
w2s = np.arange(-2, 2, 0.1)
w1z, w2z = np.meshgrid(w1s, w2s)
C = np.zeros([w1s.size, w2s.size])
for i, w1 in enumerate(w1s):
    for j, w2 in enumerate(w2s):
        C[j,i] = getLoss(W,b,w1,w2,train_y.T,train_x,w1_r,w1_c,w2_r,w2_c)
CS = plt.contour(w1z, w2z, C, cmap=cm.coolwarm)

plt.plot([a for a, b in gd_traj], [b for a,b in gd_traj], 'yo-', label="No Momentum")
plt.plot([a for a, b in mo_traj], [b for a,b in mo_traj], 'go-', label="Momentum")
plt.legend()
plt.xlabel('w1')
plt.ylabel('w2')
plt.title('Contour plot')
plt.show()
