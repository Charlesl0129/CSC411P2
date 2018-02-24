from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
from PIL import Image
import shutil

os.chdir(os.path.dirname(__file__))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if os.path.exists('CSC411P1'):
    shutil.rmtree('CSC411P1')
    
# os.mkdir('final_male')
# os.mkdir('final_female')
# os.mkdir('uncropped_female')
# os.mkdir('uncropped_male')
os.mkdir('final_male10')
os.mkdir('final_female10')
os.mkdir('uncropped_female10')
os.mkdir('uncropped_male10')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


url = 'http://www.teach.cs.toronto.edu/~csc411h/winter/projects/proj1/facescrub_actors.txt'
urllib.urlretrieve (url, 'facescrub_actors.txt')

url = 'http://www.teach.cs.toronto.edu/~csc411h/winter/projects/proj1/facescrub_actresses.txt'
urllib.urlretrieve (url, 'facescrub_actresses.txt')


#act = list(set([a.split("\t")[0] for a in open("subset_actors.txt").readlines()]))
act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.


def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()            

# #Note: you need to create the uncropped folder first in order 
# #for this to work


# #download, crop, convert to grayscale --> actresses
# for a in act:
#     name = a.split()[1].lower()
#     i = 0
#     for line in open("facescrub_actresses.txt"):
#         if a in line:
#             filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
#             timeout(testfile.retrieve, (line.split()[4], "uncropped_female/"+filename), {}, 10)
#             if not os.path.isfile("uncropped_female/"+filename):
#                 continue
    
#             print filename
#             i += 1
            
#             try:
#                 #test for validity:
#                 test = Image.open("uncropped_female/"+filename) 
#                 #read the image:
#                 im = imread ("uncropped_female/"+filename)
#                 #convert to grayscale:
#                 im_gray = rgb2gray(im)
#                 #cropping:
#                 b_box = line.split()[5]
#                 cropped_im_gray = im_gray[int(b_box.split(',')[1]):int(b_box.split(',')[3]),int(b_box.split(',')[0]):int(b_box.split(',')[2])]
#                 #resizing:
#                 final_im = imresize(cropped_im_gray,(32,32))
#                 #saving:
#                 plt.imsave("final_female/"+filename,final_im,cmap = cm.gray)
#             except Exception as e:
#                 print (e)
#                 continue
                
# #download, crop, convert to grayscale --> actors            
# for a in act:
#     name = a.split()[1].lower()
#     i = 0
#     for line in open("facescrub_actors.txt"):
#         if a in line:
#             filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
#             timeout(testfile.retrieve, (line.split()[4], "uncropped_male/"+filename), {}, 10)
#             if not os.path.isfile("uncropped_male/"+filename): 
#                 continue

#             print filename
#             i += 1
            
#             try:
#                 #test for validity:
#                 test = Image.open("uncropped_male/"+filename) 
#                 #read the image:
#                 im = imread ("uncropped_male/"+filename)
#                 #convert to grayscale:
#                 im_gray = rgb2gray(im)
#                 #cropping:
#                 b_box = line.split()[5]
#                 cropped_im_gray = im_gray[int(b_box.split(',')[1]):int(b_box.split(',')[3]),int(b_box.split(',')[0]):int(b_box.split(',')[2])]
#                 #resizing:
#                 final_im = imresize(cropped_im_gray,(32,32))
#                 #saving:
#                 plt.imsave("final_male/"+filename,final_im,cmap = cm.gray)
#             except Exception as e:
#                 print (e)
#                 continue 

# to_remove_male = ['baldwin66.jpg','carell92.jpg','hader4.jpg','hader62.jpg','hader97.jpg']
# to_remove_female = ['bracco92.jpg','harmon50.jpg']
# for bad_file in to_remove_male:
#     os.remove ('final_male/'+bad_file)
# for bad_file in to_remove_female:
#     os.remove ('final_female/'+bad_file)

                
                
                
                
#for part 10: save images as 227*227*3
for a in act:
    name = a.split()[1].lower()
    i = 0
    for line in open("facescrub_actresses.txt"):
        if a in line:
            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
            timeout(testfile.retrieve, (line.split()[4], "uncropped_female10/"+filename), {}, 10)
            if not os.path.isfile("uncropped_female10/"+filename):
                continue
    
            print filename
            i += 1
            
            try:
                #test for validity:
                test = Image.open("uncropped_female10/"+filename) 
                #read the image:
                im = imread ("uncropped_female10/"+filename)
                #cropping:
                b_box = line.split()[5]
                cropped_im = im[int(b_box.split(',')[1]):int(b_box.split(',')[3]),int(b_box.split(',')[0]):int(b_box.split(',')[2])][:,:,:3]
                #resizing:
                final_im = imresize(cropped_im,(227,227))
                #saving:
                plt.imsave("final_female10/"+filename,final_im)
            except Exception as e:
                print (e)
                continue
            
for a in act:
    name = a.split()[1].lower()
    i = 0
    for line in open("facescrub_actors.txt"):
        if a in line:
            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
            timeout(testfile.retrieve, (line.split()[4], "uncropped_male10/"+filename), {}, 10)
            if not os.path.isfile("uncropped_male10/"+filename): 
                continue

            print filename
            i += 1
            
            try:
                #test for validity:
                test = Image.open("uncropped_male10/"+filename) 
                #read the image:
                im = imread ("uncropped_male10/"+filename)
                #cropping:
                b_box = line.split()[5]
                cropped_im = im[int(b_box.split(',')[1]):int(b_box.split(',')[3]),int(b_box.split(',')[0]):int(b_box.split(',')[2])][:,:,:3]
                #resizing:
                final_im = imresize(cropped_im,(227,227))
                #saving:
                plt.imsave("final_male10/"+filename,final_im)
            except Exception as e:
                print (e)
                continue

to_remove_male = ['baldwin66.jpg','carell92.jpg','hader4.jpg','hader62.jpg','hader97.jpg']
to_remove_female = ['bracco92.jpg','harmon50.jpg']
for bad_file in to_remove_male:
    os.remove ('final_male10/'+bad_file)
for bad_file in to_remove_female:
    os.remove ('final_female10/'+bad_file


        