''' flask app with mongo '''
import os
import json
import datetime
from flask import Flask

from keras.models import Model, load_model


# create the flask object
app = Flask(__name__)

first_unet_model = None

@app.route('/')
def hello_world():
    return 'Hello, World!'


if __name__ == '__main__':
    
    # loading first unet model
    model_file = 'app/input_models/unet-carvana-augmented.hdf5'
    first_unet_model=load_model(model_file)

    app.config['DEBUG'] = 1 
    app.run(host='0.0.0.0', port=int(4000)) # Run the app
