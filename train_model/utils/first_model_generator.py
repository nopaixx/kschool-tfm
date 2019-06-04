import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from skimage.transform import resize
from keras.preprocessing.image import array_to_img, img_to_array, load_img#,save_img
from skimage.io import imread, imshow #, concatenate_images
import PIL
from PIL import Image
import re
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from .first_model_globals import H,W,BATCH_SIZE,car_path, mask_path


#### MY IMPORTS###
from .first_model_utils import downsample
from .first_model_utils import get_only_object
from .first_model_utils import augmentation

BATCH_SIZE=20
def generator_resize(path, path_backs, batch_size, h, w):
    
    while True:
        input_img=[]
        output_mask=[]
        for x in range(batch_size):
            selected = (np.random.choice(path,1))
            carimage = car_path+str(np.squeeze(selected))
            maskimage = mask_path+str(np.squeeze(selected)).replace('.jpg','_mask.gif')
            s_back=np.random.choice(path_backs,1)
            selected_back=str(np.squeeze(s_back))
            #print(selected_back)
            #print(carimage)
            back_img = imread(selected_back)/255
            img = imread(carimage)/255
            mask = imread(maskimage) / 255           
            #img, mask = augmentation(img, mask)            
            # added custom augmentation
            # img, mask = augmentation(img,mask)      
            img = downsample(img, h, w)
            mask = downsample(mask, h, w)
            back_img = downsample(back_img, h, w)
            final = get_only_object(np.uint8(img*255), np.uint8(mask*255), np.uint8(back_img*255))
            img, mask = augmentation(final,mask)
            #mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            img = cv2.cvtColor(np.uint8(img*255), cv2.COLOR_RGB2GRAY).reshape(h,w,1)
            # TODO A debug some image greate 255 others NOT...
            if np.max(img)>1:
                img =img /255
            input_img.append(img)
            output_mask.append(mask.reshape(h,w,1))
            
        yield np.array(input_img), np.array(output_mask)
