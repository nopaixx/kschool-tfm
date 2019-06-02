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
from utils import save_np_array_to_image
from utils import debug_np_to_file
from utils import get_clean_image
from utils import get_edge_v2
from utils import random_colors

from skimage.io import imread
import tensorflow as tf
# create the flask object
app = Flask(__name__)
app.debug= True
CORS(app)
H = 256
W = 256
first_unet_model = None
draw_unet_model = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


#global graph
#graph = tf.get_default_graph()
# loading first unet model
#model_file = 'app/input_models/unet-carvana-augmented.hdf5'
#first_unet_model=load_model(model_file)

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
    debug_np_to_file(img, '/tmp/original.jpg')
    # 3.- resize image to model specs 256x256
    resized_image = np.uint8(downsample(img, H, W))
    debug_np_to_file(resized_image,'/tmp/resized.jpg')
    # 4.- convert to gray scale as model input expected
    gray_image = np.uint8(to_gray_scale_image(resized_image, H, W))
    print("GRAY-->",np.max(gray_image), file=sys.stderr)
    debug_np_to_file(gray_image.reshape(H,W), '/tmp/gray.jpg')
    # 4.- Predict model
    gray_image = gray_image.reshape(H,W,1)
    # adebug need to add to elements to avoid unkonwed isssue for now...
    lista = []
    lista.append(gray_image/255)
    lista.append(gray_image/255)
   # model_file = 'app/input_models/unet-carvana-augmented.hdf5'
   # first_unet_model=load_model(model_file)
    with graph.as_default():
        ret_img = first_unet_model.predict(np.array(lista).reshape(2,H,W,1))
    # 5 .- convert to base64 image and return
    endimage = np.uint8((ret_img[0].reshape(H,W)>0.8)*255)
    debug_np_to_file(endimage, '/tmp/end.jpg')
    ret = save_np_array_to_image(endimage)
    
    print("CONVERT!!-->",ret[:80], file=sys.stderr)
    return ret 

@app.route('/apply_mask')
def apply_mask_to_image():
    """
    Input:
        -Image Original
        -Image mask (from get_model_unet())
    Output
        -Return a bitwise image applied a applied be
    """
    # print("apply_mask_endpoint", file=sys.stderr)
    image_orig = request.args.get('orig')
    image_mask = request.args.get('mask')

    # 1.- convert base64 uri o png image and save to tmp folder
    image_name_orig = convert_and_save4(image_orig)    
    image_name_mask = convert_and_save4(image_mask)
    # 2.- Load image to numpy array
    img_orig = imread(image_name_orig)
    img_mask = imread(image_name_mask)
    # debug_np_to_file(img_mask, '/tmp/mask.jpg')
    # 3.- resize image to model specs 256x256
    resized_image_orig = np.uint8(downsample(img_orig, H, W))
    resized_image_mask = np.uint8(downsample(img_mask, H, W))
    debug_np_to_file(resized_image_mask,'/tmp/resized_mask.jpg')
    debug_np_to_file(resized_image_orig,'/tmp/resized_orig.jpg')
    # 4.- perform bitwise operation to extract car appling mask
    final = get_clean_image(resized_image_orig, resized_image_mask)
    debug_np_to_file(final, '/tmp/car.jpg')
    ret = save_np_array_to_image(final)
    return ret

@app.route('/extractedges')
def get_extract_edges():
    """This endpoint return edge from image
    Input:
        -Recibe a image car (onlycar)
    Output
        -return and edges from car
    """
    car_img = request.args.get('data')
    # 1.- convert base64 to png image
    image_name = convert_and_save4(car_img)
    # 2.- Load image as numpy
    np_img = imread(image_name)
    # 3.-Apply canny edge algoritm
    np_canny = get_edge_v2(np_img)
    # 4.- transform to return b64
    ret = save_np_array_to_image(np_canny)
    return ret

@app.route('/cardraw')
def get_apply_final_model():
    """
    Final endpit
    """
    edge_image = request.args.get('data')
    # 1.- convert base64 to png image and save
    print("IMAGE-->",edge_image, file=sys.stderr)
    edge_name = convert_and_save4(edge_image)
    print("EDGE-name-->", edge_name, file=sys.stderr)
    # 2.- Load image as numpy
    np_img = imread(edge_name).reshape(H,W,1)
    print("EDGE_MAX-->", np.max(np_img), np_img.shape, file=sys.stderr)
    img = np_img/255
    # 3.- set random colors for colorize image
    color_mask = random_colors()/255
    # 4.-pred model
    lista = []
    lista.append(img)
    lista.append(img)

    lista_mask = []
    lista_mask.append(color_mask)
    lista_mask.append(color_mask)

    with graph.as_default():
        ret_img = draw_unet_model.predict([np.array(lista).reshape(2,H,W,1), np.array(lista_mask).reshape(2,H,W,3)])
   
    endimage = np.uint8((ret_img[1][0])*255)
#    debug_np_to_file(endimage, '/tmp/end.jpg')
    ret = save_np_array_to_image(endimage)
#
    
    return ret
    

if __name__ == '__main__':
    global graph
    graph = tf.get_default_graph()
    # loading first unet model
    model_file = 'app/input_models/unet-carvana-augmented.hdf5'
    model_draw = 'app/input_models/unet_standford_edges_to_image.hdf5'
    first_unet_model=load_model(model_file)
    draw_unet_model=load_model(model_draw)
    print("MASK MODEL->", first_unet_model.summary(), file=sys.stderr)

    print("DRAWMODEL->", draw_unet_model.summary(), file=sys.stderr)

    app.config['DEBUG'] = 1 

    app.run(host='0.0.0.0', port=int(4000)) # Run the app

