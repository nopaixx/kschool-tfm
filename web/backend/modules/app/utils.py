from __future__ import print_function
from base64 import decodestring
import base64
import re
import os
from io import BytesIO
import time
from PIL import ImageFile
import cv2
import numpy as np
from PIL import Image
from base64 import decodestring
from skimage.io import imread
from skimage.transform import resize
import scipy.misc
import sys
import random

image_path='/tmp/'


def debug_np_to_file(np_arr, filename):
    from PIL import Image
    im = Image.fromarray(np_arr)
    im.save(filename)
    print(filename, "-->", np_arr.shape, "-->", np.max(np_arr), file=sys.stderr)
    return None

def save_np_array_to_image(image_array):
    # cv2.imshow('image',ret_img[0])
    # ret = from_np_to_b64_image(ret_img[0])
    # ret_file  save_np_array_to_image(ret_img[0])
    # Convert captured image to JPG
    retval, buffer = cv2.imencode('.png',np.uint8(image_array))
    jpg_text = base64.b64encode(buffer)
    # ret = base64.b64encode(ret_img[0].reshape(H*W))
    # print("RET-->",ret, file=sys.stderr)

    
    return jpg_text

def convert_and_save4(b64_string):
    _, data=b64_string.split(',')
    data += '=' * (-len(data) % 4)  # restore stripped '='s
    byte_data = base64.b64decode(data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    t = time.time()
    # t = 'image'
    img.save(image_path + str(t) + '.png', "PNG")
    return image_path+str(t)+'.png'

def downsample(img, h, w):    
    ret = resize(img, (h, w), mode='constant', preserve_range=True)
    return ret


def to_gray_scale_image(img, H=256, W=256):
    img = cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2GRAY).reshape(H,W,1)
    return img

def from_np_to_b64_image(arr):

    im = Image.fromarray(arr)
    #im.show()  # uncomment to look at the image
    rawBytes = io.BytesIO()
    im.save(rawBytes, "PNG")
    rawBytes.seek(0)  # return to the start of the file
    return base64.b64encode(rawBytes.read())


def get_clean_image(img,mask):
    """Perform a bitwise operation netween orig image and mask"""
    # img = cv2.imread(path)
    # img = cv2.resize(img, (h, w))
    # print(imshow(img))
    # mask = cv2.imread(maskimage)
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

def random_colors(H=256, W=256):
    avg=np.zeros((H,W,3))
    #print(img_reduct[img_reduct[:,:,0]<255,0].mean())
    avg[...,:,0]=255
    avg[...,:,1]=255
    avg[...,:,2]=255
    return avg

