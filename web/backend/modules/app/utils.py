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

image_path='/tmp/'

def convert_and_save4(b64_string):
    _, data=b64_string.split(',')
    data += '=' * (-len(data) % 4)  # restore stripped '='s
    byte_data = base64.b64decode(data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    #t = time.time()
    t = 'image'
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
