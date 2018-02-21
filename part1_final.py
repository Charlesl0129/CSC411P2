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
import shutil

os.chdir(os.path.dirname(__file__))

if os.path.exists('figures'):
    shutil.rmtree('figures')
os.mkdir ('figures')
#Function Definition Begins
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

M = loadmat('mnist_all.mat')
train_x, train_y = get_train(M)
test_x, test_y = get_test(M)


#zeros
np.random.seed(0)
fig = plt.figure()
for i in range (1,11):
    fig.add_subplot(2,5,i)
    im = int(random.random()*5923)
    plt.imshow(train_x[im].reshape((28,28)),cmap = plt.cm.gray)
plt.savefig('figures/part1Zeros.jpg')

#ones
fig = plt.figure()

for i in range (1,11):
    fig.add_subplot(2,5,i)
    im = int(random.random()*6742+5923)
    plt.imshow(train_x[im].reshape((28,28)),cmap = plt.cm.gray)
plt.savefig('figures/part1Ones.jpg')

#twos
fig = plt.figure()
for i in range (1,11):
    fig.add_subplot(2,5,i)
    im = int(random.random()*5958+12665)
    plt.imshow(train_x[im].reshape((28,28)),cmap = plt.cm.gray)
plt.savefig('figures/part1Twos.jpg')

#threes
fig = plt.figure()
for i in range (1,11):
    fig.add_subplot(2,5,i)
    im = int(random.random()*6131+18623)
    plt.imshow(train_x[im].reshape((28,28)),cmap = plt.cm.gray)
plt.savefig('figures/part1Threes.jpg')

#fours
fig = plt.figure()
for i in range (1,11):
    fig.add_subplot(2,5,i)
    im = int(random.random()*5842+24754)
    plt.imshow(train_x[im].reshape((28,28)),cmap = plt.cm.gray)
plt.savefig('figures/part1Fours.jpg')

#fives
fig = plt.figure()
for i in range (1,11):
    fig.add_subplot(2,5,i)
    im = int(random.random()*5421+30596)
    plt.imshow(train_x[im].reshape((28,28)),cmap = plt.cm.gray)
plt.savefig('figures/part1Fives.jpg')

#sixs
fig = plt.figure()
for i in range (1,11):
    fig.add_subplot(2,5,i)
    im = int(random.random()*5918+36017)
    plt.imshow(train_x[im].reshape((28,28)),cmap = plt.cm.gray)
plt.savefig('figures/part1Sixs.jpg')

#sevens
fig = plt.figure()
for i in range (1,11):
    fig.add_subplot(2,5,i)
    im = int(random.random()*6265+41935)
    plt.imshow(train_x[im].reshape((28,28)),cmap = plt.cm.gray)
plt.savefig('figures/part1Sevens.jpg')

#eights
fig = plt.figure()
for i in range (1,11):
    fig.add_subplot(2,5,i)
    im = int(random.random()*5851+48200)
    plt.imshow(train_x[im].reshape((28,28)),cmap = plt.cm.gray)
plt.savefig('figures/part1Eights.jpg')

#nines
fig = plt.figure()
for i in range (1,11):
    fig.add_subplot(2,5,i)
    im = int(random.random()*5949+54051)
    plt.imshow(train_x[im].reshape((28,28)),cmap = plt.cm.gray)
plt.savefig('figures/part1Nines.jpg')