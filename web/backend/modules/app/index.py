from __future__ import print_function
import os
import json
import datetime
import sys
import base64
from flask import Flask
from flask import request
from keras.models import Model, load_model
from PIL import Image
from flask_cors import CORS
from base64 import decodestring
import base64
import re
import os
from io import BytesIO
import time
from PIL import ImageFile
import cv2
import numpy as np

from utils import convert_and_save4
from utils import downsample
from utils import to_gray_scale_image
from utils import from_np_to_b64_image

from skimage.io import imread
import tensorflow as tf
# create the flask object
app = Flask(__name__)
CORS(app)
H = 256
W = 256
first_unet_model = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


@app.route('/')
def hello_world():
    return 'done' 

@app.route('/extract_mask', methods=['GET', 'POST'])
def get_model_unet():
    """
    This endpoint return first model prediction car MASKING
    Input: 
        -User image
    Output
        - Return a first model prediction images
    """
    image_b64 = request.args.get('data')
    # print("--->", "AAAAA" , file=sys.stderr)
    # 1.- convert base64 uri o png image and save to tmp folder
    image_name =  convert_and_save4(image_b64)

    # 2.- Load image to numpy array
    img = imread(image_name)
    # 3.- resize image to model specs 256x256
    resized_image = downsample(img, H, W)
    # 4.- convert to gray scale as model input expected
    gray_image = to_gray_scale_image(resized_image, H, W)
    # 4.- Predict model
    gray_image = gray_image.reshape(H,W,1)
    # adebug need to add to elements to avoid unkonwed isssue for now...
    lista = []
    lista.append(gray_image/255)
    lista.append(gray_image/255)
    with graph.as_default():
        ret_img = first_unet_model.predict(np.array(lista).reshape(2,H,W,1))

#    cv2.imshow('image',ret_img[0])
    #ret = from_np_to_b64_image(ret_img[0])
    ret = base64.b64encode(ret_img[0])
#    print("RET-->",ret, file=sys.stderr)
    return ret
@app.route('/apply_mask_to_image')
def apply_mask_to_image():
    """
    Input:
        -Image Original
        -Image mask (from get_model_unet())
    Output
        -Return a bitwise image applied a applied be
    """
    return "apply mask to image"

@app.route('/extract_edges')
def get_extract_edges():
    """This endpoint return edge from image
    Input:
        -Recibe a image car (onlycar)
    Output
        -return and edges from car
    """

    return 'Extract edges'

@app.route('/final_model')
def get_apply_final_model():
    """
    Final endpit
    """
    return 'final_model'
    

if __name__ == '__main__':
    global graph
    graph = tf.get_default_graph()
    # loading first unet model
    model_file = 'app/input_models/unet-carvana-augmented.hdf5'
    first_unet_model=load_model(model_file)
    print(first_unet_model.summary(), file=sys.stderr)
    app.config['DEBUG'] = 1 

    app.run(host='0.0.0.0', port=int(4000)) # Run the app

