import os
import sys
import subprocess
import requests
import ssl
import random
import string
import json

from flask import jsonify
from flask import Flask
from flask import request
from flask import send_file
import traceback

from app_utils import blur
from app_utils import download
from app_utils import generate_random_filename
from app_utils import clean_me
from app_utils import clean_all
from app_utils import create_directory
from app_utils import get_model_bin
from app_utils import get_multi_model_bin
from app_utils import unzip
from app_utils import unrar
from app_utils import resize_img
from app_utils import square_center_crop
from app_utils import square_center_crop
from app_utils import image_crop

from keras_segmentation import pretrained




from threading import Thread
from multiprocessing import Process, Queue



try:  # Python 3.5+
    from http import HTTPStatus
except ImportError:
    try:  # Python 3
        from http import client as HTTPStatus
    except ImportError:  # Python 2
        import httplib as HTTPStatus


app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def segmentation(model, input_path, output_path):

    
    if model == "scene_parsing":
        model = pretrained.pspnet_50_ADE_20K()

    elif model == "cityscapes":
        model = pretrained.pspnet_101_cityscapes()
    else :
        model == pretrained.pspnet_101_voc12()


    model.predict_segmentation(
        inp=input_path,
        out_fname=output_path
    )


@app.route("/process", methods=["POST"])
def process():

    input_path = generate_random_filename(upload_directory,"jpg")
    output_path = generate_random_filename(result_directory,"png")

    input_path = 'input.jpg'

    try:

        if 'file' in request.files:
            file = request.files['file']
            if allowed_file(file.filename):
                file.save(input_path)

            model = request.form.getlist('model')[0]
        else:
            url = request.json["url"]
            download(url, input_path)
            model = request.json["model"]


        if prewarm:
            segmentation(model, input_path, output_path)
        else:
            p = Process(target=segmentation, args=(model, input_path, output_path))
            p.start()
            p.join() # this blocks until the process terminates    

  
        callback = send_file(output_path, mimetype='image/png')            

        return callback, 200

    except:
        traceback.print_exc()
        return {'message': 'input error'}, 400

    finally:
        clean_all([
            input_path,
            output_path
            ])

if __name__ == '__main__':
    global upload_directory
    global result_directory
    global model_scene_parsing, model_cityscapes, model_visual_object
    global ALLOWED_EXTENSIONS
    global prewarm

    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

    result_directory = '/src/results/'
    create_directory(result_directory)

    upload_directory = '/src/upload/'
    create_directory(upload_directory)

    prewarm = True if os.getenv('PREWARM', 'TRUE') == 'TRUE' else False

    if prewarm:
        model_scene_parsing = pretrained.pspnet_50_ADE_20K() # load the pretrained model trained on ADE20k dataset
        model_cityscapes= pretrained.pspnet_101_cityscapes() # load the pretrained model trained on Cityscapes dataset
        model_visual_object = pretrained.pspnet_101_voc12() # load the pretrained model trained on Pascal VOC 2012 dataset
    else:
        model_scene_parsing = None
        model_cityscapes = None
        model_visual_object = None

    port = 5000
    host = '0.0.0.0'

    app.run(host=host, port=port, threaded=False)

