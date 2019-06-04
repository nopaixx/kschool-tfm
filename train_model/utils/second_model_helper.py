
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
from skimage.transform import resize
from keras.preprocessing.image import array_to_img, img_to_array, load_img#,save_img
from skimage.io import imread, imshow #, concatenate_images
import PIL 
from PIL import Image
import re
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import math
import glob
import random

from sklearn import metrics, manifold
from matplotlib import pyplot as plt

import keras
from keras.layers import Input, Conv2DTranspose,Dense, Dropout, Conv2D, MaxPool2D, Flatten, LSTM, Conv1D, MaxPool1D, Lambda, Multiply,UpSampling2D
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.utils import np_utils 
from keras.datasets import mnist
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import sequence
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from keras import backend as K



from .first_model_globals import H,W,BATCH_SIZE


def downsample(img, h, w):    
    ret = resize(img, (h, w), mode='constant', preserve_range=True)
    # plt.imshow(ret)
    return ret
    #return cv2.resize(img, (h, w))
def get_clean_image(img,mask):    
    # img = cv2.imread(path)
    # img = cv2.resize(img, (h, w))
    # print(imshow(img))
    #mask = cv2.imread(maskimage)
    # mask=np.int8((imread(path_mask)))
    # print(mask)   
    # get first masked value (foreground)
    img = np.uint8(img)
    mask = np.uint8(mask)
    fg = cv2.bitwise_or(img, img, mask=mask)
    # get second masked value (background) mask must be inverted
    mask = cv2.bitwise_not(mask)
    background = np.full(img.shape, 255, dtype=np.uint8)
    bk = cv2.bitwise_or(background, background, mask=mask)

    # combine foreground+background
    final = cv2.bitwise_or(fg, bk)

    return final

import cv2
def get_edge_v2(img):
    blurred = cv2.GaussianBlur(img, (7,7), 0) # Remove noise
    #close the small line gaps using errosion
    kernel = np.ones((5,5), np.uint8)
    erode = cv2.erode(blurred, kernel, iterations = 1)
    #plt.imshow(erode)
    #cannyedge 
    def canny_edge_detector(input_img, threshold1, threshold2, draw=True, save=True):
        canny_img = cv2.cvtColor(np.copy(input_img), cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(canny_img, threshold1, threshold2)
        return edges
    #try adding Eroding before edge detection(increase black lines)
    canny_edges = canny_edge_detector(input_img=erode, threshold1=50, threshold2=180) 
    #return np.invert(canny_edges)
    #return np.invert(canny_edges)
    #plt.imshow(canny_edges)

    #close the small line gaps using dilation
    kernel = np.ones((3,3), np.uint8)
    dilation_canny = cv2.dilate(canny_edges, kernel, iterations = 1)
    return np.invert(dilation_canny)
    canny_blurred = cv2.GaussianBlur(dilation_canny, (3,3), 0) # Remove noise
    #invetimos la mascara queremos el blanco de fondo y el negro como dibujo del coche
    return np.invert(canny_blurred)

def get_avg_color_mask(img_reduct):
    avg=np.ones((H,W,3))
    #print(img_reduct[img_reduct[:,:,0]<255,0].mean())
    avg[...,:,0]=img_reduct[img_reduct[:,:,0]<255,0].mean()
    avg[...,:,1]=img_reduct[img_reduct[:,:,1]<255,1].mean()
    avg[...,:,2]=img_reduct[img_reduct[:,:,2]<255,2].mean()
    return avg

def get_img_info(img, model, graph):
    #plt.imshow(img)
    #print("SHAPE-->", img.shape)
    #plt.show()
    lista=[]
    tmp = downsample(img,H,W)
    
    img = cv2.cvtColor(np.uint8(tmp), cv2.COLOR_RGB2GRAY).reshape(H,W,1)
    lista.append(img/255)
    lista.append(img/255)
    with graph.as_default():
        xpred = model.predict(np.array(lista).reshape(2,H,W,1))
    #plt.imshow(xpred[0].reshape(H,W))
    #plt.show()
    #print(np.max(img))
    final_car = get_clean_image(tmp, ((xpred[0]>0.5)*255))
    edge = get_edge_v2(np.uint8(tmp))
    #plt.imshow(final_car)
    #plt.show()
    avg = get_avg_color_mask(final_car)/255
    return img.reshape(H,W)/255, xpred[0].reshape(H,W), final_car/255, edge/255, avg

