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
from base64 import decodestring
import base64
import re
import os
from io import BytesIO
import time
from PIL import ImageFile
import cv2
import numpy as np


# create the flask object
app = Flask(__name__)

first_unet_model = None
image_path='/tmp/'
ImageFile.LOAD_TRUNCATED_IMAGES = True
def convert_and_save5(b64_string, image_name):  
    _, data=b64_string.split(',')
    # data += '=' * (-len(data) % 4)  # restore stripped '='s
    data = data.replace(' ','').replace('\n','').replace('\r','').strip()
    data += '=' * (-len(data) % 4)  # restore stripped '='s
    imgdata = base64.b64decode(data)
    with open(image_name, 'wb') as f:
        f.write(imgdata)

def convert_and_save4(b64_string, image_name):
#    def getI420FromBase64(codec, image_path=""):
    # get codec from http
    # codec = codec.encode()

#    b64_string += '=' * (-len(b64_string) % 4)  # restore stripped '='s
   # data = re.sub('^data:image/.+;base64,', '', b64_string)
    _,data=b64_string.split(',')
    print("AAAAA--->", _, file=sys.stderr)
 #   data = data.replace(' ','').replace('\n','').replace('\r','').strip()
    data += '=' * (-len(data) % 4)  # restore stripped '='s
#    byte_data = base64.encodestring(data)
    im = Image.open(BytesIO(base64.b64decode(data)))
    #print("AAAAA--->", data, file=sys.stderr)
 #   image_data = BytesIO(byte_data)
    #img = Image.open(image_data)
    #t = time.time()
    # t = 'image'
    # img.save(image_path + str(t) + '.png', "PNG")
    im.save('/tmp/image.png', "PNG")
#    nparr = np.fromstring(data.decode('base64'), np.uint8)
#    img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
#    cv2.imwrite('/tmp/image.png', img)



def convert_and_save2(b64_string, image_name):

    b64_string += '=' * (-len(b64_string) % 4)  # restore stripped '='s

    string = b'{b64_string}'

    with open(image_name, "wb") as fh:
        fh.write(base64.b64decode(b64_string.encode()))

def convert_and_save3(b64_string, image_name):
    b64_string += '=' * (-len(b64_string) % 4)  # restore stripped '='s
    image = Image.fromstring('RGB',(256, 256),base64.b64decode(b64_string.encode()))
    image.save(image_name)

def convert_and_save(b64_string, image_name):
    b64_string += '=' * (-len(b64_string) % 4)  # restore stripped '='s
    image = Image.fromstring('RGB',base64.b64decode(b64_string.encode()))
    image.save(image_name)
    # with open(image_name, "wb") as fh:
    #     fh.write(decodebytes(b64_string.encode()))

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
    print("--->", "AAAAA" , file=sys.stderr)
   # print("--->",image_b64[0:100], file=sys.stderr)
    # image_b64 += "=" * ((4 - len(image_b64) % 4) % 4) #ugh
    # image_b64 = request.data

#    print(image_b64, file=sys.stderr)
    convert_and_save5(image_b64, '/tmp/image1.png')
#   data = request.get_json()
#    img_data = data['img']
    # this method convert and save the base64 string to image
#    convert_and_save(img_data,image_name)
    
    # once we have on file then we can opened
    # load image in memory
    # 
    return 'Unet Model'

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
    
    # loading first unet model
    model_file = 'app/input_models/unet-carvana-augmented.hdf5'
    first_unet_model=load_model(model_file)

    app.config['DEBUG'] = 1 
    app.run(host='0.0.0.0', port=int(4000)) # Run the app

