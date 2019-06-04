

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
import cv2


from .first_model_globals import H,W,BATCH_SIZE
from .second_model_helper import get_img_info

def generator_for_standford(paths, batch_size, h, w, model, graph):
    
    while True:
        edges_in=[]
        color_mask_in=[]

        bw_img_out=[]
        end_img_out=[]
        for x in range(batch_size):
            randomimg = np.random.choice(paths,1)
            img = imread(randomimg[0])
            # a few images on dataset are black white with dimension 2 and not valid
            while True:
                if (img.ndim == 3) & (img.shape[0]>256):
                    #print("break")
                    break
                #print(img.ndim, img.shape)
                randomimg = np.random.choice(paths,1)
                img = imread(randomimg[0])
            resized, mask, end_car, edges, avg = get_img_info(img, model, graph)
            img_bw = cv2.cvtColor(np.uint8(end_car*255), cv2.COLOR_RGB2GRAY)
            edges_in.append(edges.reshape(h,w,1))
            color_mask_in.append(avg)
            bw_img_out.append(img_bw.reshape(h,w,1)/255)
            end_img_out.append(end_car)
        
        yield [np.array(edges_in),np.array(color_mask_in)], [np.array(bw_img_out),np.array(end_img_out)]
