# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import numpy as np
import pandas as pd
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
from sklearn.model_selection import train_test_split

##############IMPORT MY UTILS#####################

from utils.first_model_utils import intel_files_names_in_path
from utils.first_model_utils import files_names_in_path_carvana
from utils.first_model_generator import generator_resize
from utils.first_model_keras import get_unet

################END MY UTILS#######################

from utils.first_model_globals import H,W,BATCH_SIZE,car_path, mask_path

###########LOADING INTEL IMAGE BACKGROUND##########
building = '../input/intel-image-classification/seg_train/seg_train/buildings/'
street = '../input/intel-image-classification/seg_train/seg_train/street/'
background_path = intel_files_names_in_path(street)
for x in intel_files_names_in_path(building):
    background_path.append(x)


##################################################


########LOADING CARVANA IMAGES AND MASK###########
input_files,input_masks = files_names_in_path_carvana(mask_path)

#################################################


train_input, test_input, _, _ = train_test_split(input_files, input_files, test_size=0.15, random_state=42)

input_img = Input((H, W, 1), name='img')
model = get_unet(input_img, n_filters=64, dropout=0.5, batchnorm=True)

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=["accuracy","binary_crossentropy"])
model.summary()

#definimos nuestro callback para guardar
saver = ModelCheckpoint('../output/unet-carvana-augmented.hdf5', save_best_only=True, monitor='val_loss', mode='min')

histo = model.fit_generator(generator_resize(train_input,background_path,BATCH_SIZE, H, W),steps_per_epoch=10, epochs=75,
                    validation_data=generator_resize(test_input,background_path,BATCH_SIZE, H, W),validation_steps=10,
                    callbacks=[saver])



