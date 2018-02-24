from pylab import *
from torch.autograd import Variable
import torch
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import *
import os
from scipy.io import loadmat
from random import shuffle

np.random.seed(7)
torch.manual_seed(7)
os.chdir(os.path.dirname(__file__))

#fetching the files names:
filelist_male = []
filelist_female = []
for filename in os.walk('final_female'):
    filelist_female.append(filename)
for filename in os.walk('final_male'):
    filelist_male.append(filename)
filelist_male = filelist_male[0][2]
filelist_female = filelist_female [0][2]

#getting all images of actors:
baldwin_list = []
carell_list = []
hader_list = []
for file in filelist_male:
    if 'baldwin' in file:
        baldwin_list.append(file)
    elif 'carell' in file:
        carell_list.append(file)
    elif 'hader' in file:
        hader_list.append(file)

#getting all images of actresses:
bracco_list = []
gilpin_list = []
harmon_list = []
for file in filelist_female:
    if 'bracco' in file:
        bracco_list.append(file)
    elif 'gilpin' in file:
        gilpin_list.append(file)
    elif 'harmon' in file:
        harmon_list.append(file)
        
#shuffle the lists for randomness:
lists = [baldwin_list,carell_list,hader_list,bracco_list,gilpin_list,harmon_list]
for l in lists:
    np.random.shuffle(l)

#initialize training sets:
training_baldwin = []
training_carell = []
training_hader = []
training_bracco = []
training_gilpin = []
training_harmon = []

#initialize validation sets:
validation_baldwin = []
validation_carell = []
validation_hader = []
validation_bracco = []
validation_gilpin = []
validation_harmon = []

#initialize test sets:
test_baldwin = []
test_carell = []
test_hader = []
test_bracco = []
test_gilpin = []
test_harmon = []

#forming training sets:
for i in range (45):
    training_baldwin.append(baldwin_list.pop())
    training_carell.append(carell_list.pop())
    training_hader.append(hader_list.pop())
    training_bracco.append(bracco_list.pop())
    training_harmon.append(harmon_list.pop())
    training_gilpin.append(gilpin_list.pop())

#forming validation sets:
for i in range (20):
    validation_baldwin.append(baldwin_list.pop())
    validation_carell.append(carell_list.pop())
    validation_hader.append(hader_list.pop())
    validation_bracco.append(bracco_list.pop())
    validation_gilpin.append(gilpin_list.pop())
    validation_harmon.append(harmon_list.pop())

#forming test sets:
for i in range (20):
     test_baldwin.append(baldwin_list.pop())
     test_carell.append(carell_list.pop())
     test_hader.append(hader_list.pop())
     test_bracco.append(bracco_list.pop())
     test_gilpin.append(gilpin_list.pop())
     test_harmon.append(harmon_list.pop())

x = ones ((1,1024))
y_train = array([0.,0.,0.,0.,0.,0.])
y_test = array([0.,0.,0.,0.,0.,0.])
y_validation = array([0.,0.,0.,0.,0.,0.])
for i in range (45):
    y_train = vstack((y_train,[1.,0.,0.,0.,0.,0.])) #bracco
for i in range (45):
    y_train = vstack((y_train,[0.,1.,0.,0.,0.,0.]))  #gilpin  
for i in range (45):
    y_train = vstack((y_train,[0.,0.,1.,0.,0.,0.]))  #harmon
for i in range (45):
    y_train = vstack((y_train,[0.,0.,0.,1.,0.,0.]))  #baldwin
for i in range (45):
    y_train = vstack((y_train,[0.,0.,0.,0.,1.,0.]))  #hader
for i in range (45):
    y_train = vstack((y_train,[0.,0.,0.,0.,0.,1.]))  #carell
train_y = delete(y_train,0,0)
#print (train_y)
#print (train_y.shape)

for i in range (20):
    y_test = vstack((y_test,[1.,0.,0.,0.,0.,0.])) #bracco
    y_validation = vstack((y_validation,[1.,0.,0.,0.,0.,0.])) #bracco
for i in range (20):
    y_test = vstack((y_test,[0.,1.,0.,0.,0.,0.]))  #gilpin
    y_validation = vstack((y_validation,[0.,1.,0.,0.,0.,0.]))  #gilpin   
for i in range (20):
    y_test = vstack((y_test,[0.,0.,1.,0.,0.,0.]))  #harmon
    y_validation = vstack((y_validation,[0.,0.,1.,0.,0.,0.]))  #harmon
for i in range (20):
    y_test = vstack((y_test,[0.,0.,0.,1.,0.,0.]))  #baldwin
    y_validation = vstack((y_validation,[0.,0.,0.,1.,0.,0.]))  #baldwin
for i in range (20):
    y_test = vstack((y_test,[0.,0.,0.,0.,1.,0.]))  #hader
    y_validation = vstack((y_validation,[0.,0.,0.,0.,1.,0.]))  #hader
for i in range (20):
    y_test = vstack((y_test,[0.,0.,0.,0.,0.,1.]))  #carell
    y_validation = vstack((y_validation,[0.,0.,0.,0.,0.,1.]))  #carell

test_y = delete(y_test,0,0)
validation_y = delete(y_validation,0,0)
# print (test_y.shape)
# print (validation_y.shape)


#initializing big X matrix: using 65 images each actor/actresses for training
#(IMPORTANT) ordering: bracco - gilpin - harmon - baldwin - hader - carell

#loading all actresses images into X:
training_females = [training_bracco,training_gilpin,training_harmon] 
for female in training_females:
    while len(female)!= 0:
        toRead = female.pop()
        im = imread ('final_female/'+toRead)
        im = im[:,:,0]/255.
        im_1d = im.flatten()
        x = vstack((x,im_1d))


# #loading all actors images into X:
training_males = [training_baldwin,training_hader,training_carell] 
for male in training_males:
    while len(male)!= 0:
        toRead = male.pop()
        im = imread ('final_male/'+toRead)
        im = im[:,:,0]/255.
        im_1d = im.flatten()
        x = vstack((x,im_1d))

train_x = delete(x,0,0)
#print (train_x.shape)

#loading all actresses images into X:
x = ones ((1,1024))
test_females = [test_bracco,test_gilpin,test_harmon]
for female in test_females:
    while len(female)!= 0:
        toRead = female.pop()
        im = imread ('final_female/'+toRead)
        im = im[:,:,0]/255.
        im_1d = im.flatten()
        x = vstack((x,im_1d))

# #loading all actors images into X:
test_males = [test_baldwin,test_hader,test_carell]
for male in test_males:
    while len(male)!= 0:
        toRead = male.pop()
        im = imread ('final_male/'+toRead)
        im = im[:,:,0]/255.
        im_1d = im.flatten()
        x = vstack((x,im_1d))
test_x = delete(x,0,0)
#print (test_x.shape)

#loading all actresses images into X:
x = ones ((1,1024))
validation_females = [validation_bracco,validation_gilpin,validation_harmon]
for female in validation_females:
    while len(female)!= 0:
        toRead = female.pop()
        im = imread ('final_female/'+toRead)
        im = im[:,:,0]/255.
        im_1d = im.flatten()
        x = vstack((x,im_1d))

# #loading all actors images into X:
validation_males = [validation_baldwin,validation_hader,validation_carell]
for male in validation_males:
    while len(male)!= 0:
        toRead = male.pop()
        im = imread ('final_male/'+toRead)
        im = im[:,:,0]/255.
        im_1d = im.flatten()
        x = vstack((x,im_1d))
validation_x = delete(x,0,0)

#--------------------- SET UP COMPLETE ----------------------
iterList = []
perfTrainList = []
perfTestList = []
perfValidList = []

dim_x = 32*32
dim_h = 30
dim_out = 6
dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor
batch_size = 90

#minibatch
train_idx = np.random.permutation(range(train_x.shape[0]))[:1000]

model = torch.nn.Sequential(
    torch.nn.Linear(dim_x, dim_h),
    torch.nn.Tanh(),
    torch.nn.Linear(dim_h, dim_out),
    torch.nn.Softmax(),
)

loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 0.1
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9)


for i in range(400):
    train_idx_full = np.random.permutation(range(270))
    for j in range(270/batch_size):
        x = Variable(torch.from_numpy(train_x[train_idx_full[batch_size*(j-1):batch_size*j-1]]), requires_grad=False).type(dtype_float)
        #print (train_y[train_idx_full[30*(j-1):30*j-1]])
        y_classes = Variable(torch.from_numpy(np.argmax(train_y[train_idx_full[batch_size*(j-1):batch_size*j-1]], 1)), requires_grad=False).type(dtype_long)
        
        y_pred = model(x)
        loss = loss_fn(y_pred, y_classes)
        model.zero_grad()  # Zero out the previous gradient computation
        loss.backward()    # Compute the gradient
        optimizer.step()   # Use the gradient information to 
                       # make a step
                       

    if (i+1) % 5 == 0:
        print 'Iteration:', i+1
        print 'Loss:', loss.data    
        print '\n'  

top_weight = model[2].weight.data.numpy()
counter = 0
for i in range(30):
    if (top_weight[0][i])>0:
        plt.imshow(model[0].weight.data.numpy()[i].reshape((32, 32)), cmap=cm.coolwarm) #bracco
        plt.savefig('figures/part9bracco'+str(counter)+'.jpg') 
        counter+=1
counter = 0
for i in range(30):
    if (top_weight[1][i])>0:
        plt.imshow(model[0].weight.data.numpy()[i].reshape((32, 32)), cmap=cm.coolwarm) #gilpin
        plt.savefig('figures/part9gilpin'+str(counter)+'.jpg')  
        counter += 1
#most useful hidden
'''
top_weight = model[2].weight.data.numpy()
most_h_actor = []
for i in range(6):
    most_h_actor.append(np.argmax(top_weight[i], 0))

    

    
#part 9
directory = 'figures'
if not os.path.exists(directory):
    os.makedirs(directory) 
plt.imshow(model[0].weight.data.numpy()[most_h_actor[0]].reshape((32, 32)), cmap=cm.coolwarm) #bracco
plt.savefig('figures/part9bracco.jpg')
plt.imshow(model[0].weight.data.numpy()[most_h_actor[1]].reshape((32, 32)), cmap=cm.coolwarm) #gilpin
plt.savefig('figures/part9gilpin.jpg')
plt.imshow(model[0].weight.data.numpy()[most_h_actor[2]].reshape((32, 32)), cmap=cm.coolwarm) #harmon
plt.savefig('figures/part9harmon.jpg')
plt.imshow(model[0].weight.data.numpy()[most_h_actor[3]].reshape((32, 32)), cmap=cm.coolwarm) #baldwin
plt.savefig('figures/part9baldwin.jpg')
plt.imshow(model[0].weight.data.numpy()[most_h_actor[4]].reshape((32, 32)), cmap=cm.coolwarm) #harder
plt.savefig('figures/part9harder.jpg')
plt.imshow(model[0].weight.data.numpy()[most_h_actor[5]].reshape((32, 32)), cmap=cm.coolwarm) #carell
plt.savefig('figures/part9carell.jpg')
'''
    

    
