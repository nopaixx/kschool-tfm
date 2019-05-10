''' flask app with mongo '''
import os
import json
import datetime
from flask import Flask
from flask import request
from keras.models import Model, load_model


# create the flask object
app = Flask(__name__)

first_unet_model = None


def convert_and_save(b64_string, image_name):
    with open(image_name, "wb") as fh:
        fh.write(base64.decodebytes(b64_string.encode()))

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/extract_mask', methods=['GET', 'POST']):
def get_model_unet():
    """
    This endpoint return first model prediction car MASKING
    Input: 
        -User image
    Output
        - Return a first model prediction images
    """

    data = request.get_json()
    img_data = data['img']
    # this method convert and save the base64 string to image
    convert_and_save(img_data,image_name)
    
    # once we have on file then we can opened
    # load image in memory
    # 
    return 'Unet Model'

@app.route('/apply_mask_to_image'):
def apply_mask_to_image():
    """
    Input:
        -Image Original
        -Image mask (from get_model_unet())
    Output
        -Return a bitwise image applied a applied be
    """
    return "apply mask to image"

@app.route('/extract_edges'):
def get_extract_edges():
    """This endpoint return edge from image
    Input:
        -Recibe a image car (onlycar)
    Output
        -return and edges from car
    """

    return 'Extract edges'

@app.route('/'):
def get_apply_final_model():
    """
    Final endpit
    """

    






if __name__ == '__main__':
    
    # loading first unet model
    model_file = 'app/input_models/unet-carvana-augmented.hdf5'
    first_unet_model=load_model(model_file)

    app.config['DEBUG'] = 1 
    app.run(host='0.0.0.0', port=int(4000)) # Run the app

