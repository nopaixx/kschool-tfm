
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
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

from utils.first_model_globals import H,W,BATCH_SIZE
from utils.second_model_keras import get_unet
from utils.second_model_generator import generator_for_standford


model_file = '../output/unet-carvana-augmented.hdf5'
model=load_model(model_file)
model.summary()
standfordcars_train = '../input/stanford-car-dataset-by-classes-folder/car_data/car_data/train'
standfordcars_test = '../input/stanford-car-dataset-by-classes-folder/car_data/car_data/test'


global graph
graph = tf.get_default_graph()



images=[]

############# from directory to memory files names###############

for x in os.listdir(standfordcars_train):
    for image in os.listdir(standfordcars_train+'/'+str(x)):
        extension = os.path.splitext(image)[1]        
        if (extension=='.jpg'):
            images.append(standfordcars_train+'/'+str(x)+'/'+image)
        
for x in os.listdir(standfordcars_test):
    for image in os.listdir(standfordcars_test+'/'+str(x)):
        extension = os.path.splitext(image)[1]
        if (extension=='.jpg'):
            images.append(standfordcars_test+'/'+str(x)+'/'+image)
        

##################################################################


from sklearn.model_selection import train_test_split
train_input, test_input, _, _ = train_test_split(images, images, test_size=0.15, random_state=42)


################ MY CUSTOM UNET APROX CONECT TO GRAPH#################
input_edge = Input(shape=(256, 256, 1), name="edge")  # adapt this if using `channels_first` image data format
input_color = Input(shape=(256, 256, 3), name="color")  # adapt this if using `channels_first` image data format

outedge = get_unet(input_edge, n_filters=64, dropout=0.5, batchnorm=True,chanels_out=1)

subtracted = keras.layers.Subtract()([input_color, outedge])
# mult = keras.layers.multiply([input_color, outedge])
colorcar = get_unet(subtracted, n_filters=64, dropout=0.5, batchnorm=True,chanels_out=3)

model_std = Model([input_edge, input_color],[outedge, colorcar])
model_std.compile(optimizer='Adadelta', loss=['binary_crossentropy','binary_crossentropy'])
              #, loss_weights=[100, 1])
model_std.summary()

######################################################################


##################TRAIN MODEL###############################


# initialize the number of epochs and batch size
# definimos nuestro callback para guardar
saver = ModelCheckpoint('../output/unet_standford_edges_to_image.hdf5', save_best_only=True, monitor='val_loss', mode='min')

EPOCHS = 300
BS = 10

# train the network
histor=model_std.fit_generator(generator_for_standford(train_input,BS,H,W, model, graph),
    validation_data=(generator_for_standford(test_input,BS,H,W, model, graph)), steps_per_epoch=15, 
    validation_steps = 10,
    epochs=EPOCHS, callbacks=[saver])


#############################################################







