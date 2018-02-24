from pylab import *
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import torch
import torchvision.models as models
import torchvision
from torch.autograd import Variable
import time
import numpy as np
import  matplotlib.pyplot as plt
from scipy.misc import imread, imresize
import os
import torch.nn as nn
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
from scipy.io import loadmat

class MyAlexNet(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)
        
        features_weight_i = [0, 3, 6, 8, 10]
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias
        
    def __init__(self, num_classes=1000):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        self.load_weights()
        

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        return x

# model_orig = torchvision.models.alexnet(pretrained=True)
os.chdir(os.path.dirname(__file__))
np.random.seed(7)
torch.manual_seed(7)

model = MyAlexNet()
model.eval()




#fetching the files names:
filelist_male = []
filelist_female = []
for filename in os.walk('final_female10'):
    filelist_female.append(filename)
for filename in os.walk('final_male10'):
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
x = ones ((1,256*6*6))
training_females = [training_bracco,training_gilpin,training_harmon] 
for female in training_females:
    while len(female)!= 0:
        toRead = female.pop()
        im = imread ('final_female10/'+toRead)[:,:,:3]
        im = im - np.mean(im.flatten())
        im = im/np.max(np.abs(im.flatten()))
        im = np.rollaxis(im, -1).astype(np.float32)
        im_v = Variable(torch.from_numpy(im).unsqueeze_(0), requires_grad=False)
        
        input_from_alexnet = model.forward(im_v)[0]
        
        #print(input_from_alexnet.shape)
        input_from_alexnet.data.numpy().flatten() 
        x = vstack((x,input_from_alexnet.data.numpy().flatten()))


# #loading all actors images into X:
training_males = [training_baldwin,training_hader,training_carell] 
for male in training_males:
    while len(male)!= 0:
        toRead = male.pop()
        im = imread ('final_male10/'+toRead)[:,:,:3]
        im = im - np.mean(im.flatten())
        im = im/np.max(np.abs(im.flatten()))
        im = np.rollaxis(im, -1).astype(np.float32)
        im_v = Variable(torch.from_numpy(im).unsqueeze_(0), requires_grad=False)    
        input_from_alexnet = model.forward(im_v)[0]
        input_from_alexnet.data.numpy().flatten() 
        x = vstack((x,input_from_alexnet.data.numpy().flatten()))

train_x = delete(x,0,0)
#print (train_x.shape)

#loading all actresses images into X:
x = ones ((1,256*6*6))
test_females = [test_bracco,test_gilpin,test_harmon]
for female in test_females:
    while len(female)!= 0:
        toRead = female.pop()
        im = imread ('final_female10/'+toRead)[:,:,:3]
        im = im - np.mean(im.flatten())
        im = im/np.max(np.abs(im.flatten()))
        im = np.rollaxis(im, -1).astype(np.float32)
        im_v = Variable(torch.from_numpy(im).unsqueeze_(0), requires_grad=False)    
        input_from_alexnet = model.forward(im_v)[0]
        input_from_alexnet.data.numpy().flatten() 
        x = vstack((x,input_from_alexnet.data.numpy().flatten()))

# #loading all actors images into X:
test_males = [test_baldwin,test_hader,test_carell]
for male in test_males:
    while len(male)!= 0:
        toRead = male.pop()
        im = imread ('final_male10/'+toRead)[:,:,:3]
        im = im - np.mean(im.flatten())
        im = im/np.max(np.abs(im.flatten()))
        im = np.rollaxis(im, -1).astype(np.float32)
        im_v = Variable(torch.from_numpy(im).unsqueeze_(0), requires_grad=False)    
        input_from_alexnet = model.forward(im_v)[0]
        input_from_alexnet.data.numpy().flatten() 
        x = vstack((x,input_from_alexnet.data.numpy().flatten()))
test_x = delete(x,0,0)
#print (test_x.shape)

#loading all actresses images into X:
x = ones ((1,256*6*6))
validation_females = [validation_bracco,validation_gilpin,validation_harmon]
for female in validation_females:
    while len(female)!= 0:
        toRead = female.pop()
        im = imread ('final_female10/'+toRead)[:,:,:3]
        im = im - np.mean(im.flatten())
        im = im/np.max(np.abs(im.flatten()))
        im = np.rollaxis(im, -1).astype(np.float32)
        im_v = Variable(torch.from_numpy(im).unsqueeze_(0), requires_grad=False)    
        input_from_alexnet = model.forward(im_v)[0]
        input_from_alexnet.data.numpy().flatten() 
        x = vstack((x,input_from_alexnet.data.numpy().flatten()))

# #loading all actors images into X:
validation_males = [validation_baldwin,validation_hader,validation_carell]
for male in validation_males:
    while len(male)!= 0:
        toRead = male.pop()
        im = imread ('final_male10/'+toRead)[:,:,:3]
        im = im - np.mean(im.flatten())
        im = im/np.max(np.abs(im.flatten()))
        im = np.rollaxis(im, -1).astype(np.float32)
        im_v = Variable(torch.from_numpy(im).unsqueeze_(0), requires_grad=False)    
        input_from_alexnet = model.forward(im_v)[0]
        input_from_alexnet.data.numpy().flatten() 
        x = vstack((x,input_from_alexnet.data.numpy().flatten()))
validation_x = delete(x,0,0)

#--------------------- SET UP COMPLETE ----------------------
iterList = []
perfTrainList = []
perfTestList = []
perfValidList = []

dim_x = 256*6*6
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
                       

    if (i+1) % 1 == 0:
        print 'Iteration:', i+1
        print 'Loss:', loss.data    
        print '\n'  

        iterList.append(i+1)
        #Performance Evaluation:
        x = Variable(torch.from_numpy(train_x),requires_grad=False).type(dtype_float)
        y_pred = model(x).data.numpy()
        perfTrain = np.mean(np.argmax(y_pred, 1) == np.argmax(train_y, 1))
        perfTrainList.append(perfTrain)
        print(perfTrain)

        x = Variable(torch.from_numpy(validation_x),requires_grad=False).type(dtype_float)
        y_pred = model(x).data.numpy()
        perfValid = np.mean(np.argmax(y_pred, 1) == np.argmax(validation_y, 1))
        perfValidList.append(perfValid)
        print(perfValid)
        
        x = Variable(torch.from_numpy(test_x),requires_grad=False).type(dtype_float)
        y_pred = model(x).data.numpy()
        perfTest = np.mean(np.argmax(y_pred, 1) == np.argmax(test_y, 1))
        perfTestList.append(perfTest)
        print(perfTest)
print(iterList)
print(perfTrainList)
print(perfValidList)
print(perfTestList)
plt.plot(iterList,perfTrainList,label = 'Training',color ='b')
plt.plot(iterList,perfValidList,label = 'Validation',color ='g')
plt.plot(iterList,perfTestList,label = 'Test',color ='r')
plt.xlabel('# of Iterations')
plt.ylabel('Performance (%)')
plt.title('Performance on the Training, Validation, and Test Sets')
plt.ylim ((0.4,1.1))
plt.legend()
directory = 'figures'
if not os.path.exists(directory):
    os.makedirs(directory)   
plt.savefig('figures/part10.jpg')
plt.show()
